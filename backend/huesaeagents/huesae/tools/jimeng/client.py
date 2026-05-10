"""即梦AI API客户端"""
import os
import json
import time
import datetime
import hashlib
import hmac
from urllib.parse import quote
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()


class JimengAIError(Exception):
    """即梦AI异常"""

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")


def norm_query(params: dict) -> str:
    """规范化查询参数"""
    query = ""
    for key in sorted(params.keys()):
        if type(params[key]) == list:
            for k in params[key]:
                query = (
                    query + quote(key, safe="-_.~") + "=" + quote(k, safe="-_.~") + "&"
                )
        else:
            query = (
                query + quote(key, safe="-_.~") + "=" + quote(params[key], safe="-_.~") + "&"
            )
    query = query[:-1]
    return query.replace("+", "%20")


def hmac_sha256(key: bytes, content: str) -> bytes:
    """HMAC-SHA256加密"""
    return hmac.new(key, content.encode("utf-8"), hashlib.sha256).digest()


def hash_sha256(content: str) -> str:
    """SHA256哈希"""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def utc_now() -> datetime.datetime:
    """获取当前UTC时间"""
    try:
        from datetime import timezone

        return datetime.datetime.now(timezone.utc)
    except ImportError:

        class UTC(datetime.tzinfo):
            def utcoffset(self, dt):
                return datetime.timedelta(0)

            def tzname(self, dt):
                return "UTC"

            def dst(self, dt):
                return datetime.timedelta(0)

        return datetime.datetime.now(UTC())


class JimengClient:
    """即梦AI API客户端（单例）"""

    BASE_URL = "https://visual.volcengineapi.com"
    VERSION = "2022-08-31"
    REGION = "cn-north-1"
    SERVICE = "cv"
    CONTENT_TYPE = "application/json"
    _instance: Optional["JimengClient"] = None

    def __new__(cls, ak: str, sk: str):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, ak: str, sk: str):
        """
        初始化即梦AI客户端

        Args:
            ak: Access Key ID
            sk: Secret Access Key
        """
        if self._initialized:
            return

        self.ak = ak
        self.sk = sk
        self._initialized = True

    def _sign_request(
        self, method: str, action: str, body: dict | None = None
    ) -> dict:
        """
        发送带签名的请求

        Args:
            method: HTTP方法 (GET, POST)
            action: API Action名称
            body: 请求体

        Returns:
            响应JSON
        """
        now = utc_now()

        query = {"Action": action, "Version": self.VERSION}
        body_str = json.dumps(body) if body else ""

        credential = {
            "access_key_id": self.ak,
            "secret_access_key": self.sk,
            "service": self.SERVICE,
            "region": self.REGION,
        }

        request_param = {
            "body": body_str,
            "host": self.BASE_URL.replace("https://", ""),
            "path": "/",
            "method": method,
            "content_type": self.CONTENT_TYPE,
            "date": now,
            "query": query,
        }

        if not body_str:
            request_param["body"] = ""

        x_date = request_param["date"].strftime("%Y%m%dT%H%M%SZ")
        short_x_date = x_date[:8]
        x_content_sha256 = hash_sha256(request_param["body"])

        sign_result = {
            "Host": request_param["host"],
            "X-Content-Sha256": x_content_sha256,
            "X-Date": x_date,
            "Content-Type": request_param["content_type"],
        }

        signed_headers_str = ";".join(
            ["content-type", "host", "x-content-sha256", "x-date"]
        )

        canonical_request_str = "\n".join(
            [
                request_param["method"].upper(),
                request_param["path"],
                norm_query(request_param["query"]),
                "\n".join(
                    [
                        "content-type:" + request_param["content_type"],
                        "host:" + request_param["host"],
                        "x-content-sha256:" + x_content_sha256,
                        "x-date:" + x_date,
                    ]
                ),
                "",
                signed_headers_str,
                x_content_sha256,
            ]
        )

        hashed_canonical_request = hash_sha256(canonical_request_str)
        credential_scope = "/".join(
            [short_x_date, credential["region"], credential["service"], "request"]
        )
        string_to_sign = "\n".join(
            ["HMAC-SHA256", x_date, credential_scope, hashed_canonical_request]
        )

        k_date = hmac_sha256(
            credential["secret_access_key"].encode("utf-8"), short_x_date
        )
        k_region = hmac_sha256(k_date, credential["region"])
        k_service = hmac_sha256(k_region, credential["service"])
        k_signing = hmac_sha256(k_service, "request")
        signature = hmac_sha256(k_signing, string_to_sign).hex()

        sign_result[
            "Authorization"
        ] = "HMAC-SHA256 Credential={}, SignedHeaders={}, Signature={}".format(
            credential["access_key_id"] + "/" + credential_scope,
            signed_headers_str,
            signature,
        )

        header = {**sign_result}

        url = f"{self.BASE_URL}?Action={action}&Version={self.VERSION}"

        if method == "POST":
            response = requests.post(
                url, headers=header, data=body_str, timeout=30
            )
        else:
            response = requests.get(url, headers=header, timeout=30)

        return response.json()

    def submit_task(
        self,
        prompt: str,
        width: int = 2048,
        height: int = 2048,
        scale: float = 0.5,
        force_single: bool = True,
        image_urls: Optional[list[str]] = None,
    ) -> str:
        """
        提交文生图任务

        Args:
            prompt: 文本提示词
            width: 图片宽度，默认2048
            height: 图片高度，默认2048
            scale: 文本影响程度0-1，默认0.5
            force_single: 是否强制生成单图，默认True
            image_urls: 参考图片URL列表，可选

        Returns:
            task_id: 任务ID

        Raises:
            JimengAIError: 提交失败时抛出
        """
        body = {
            "req_key": "jimeng_t2i_v40",
            "prompt": prompt,
            "width": width,
            "height": height,
            "scale": scale,
            "force_single": force_single,
        }

        if image_urls:
            body["image_urls"] = image_urls

        response = self._sign_request("POST", "CVSync2AsyncSubmitTask", body)

        if response.get("code") != 10000:
            raise JimengAIError(
                response.get("code", -1), response.get("message", "Unknown error")
            )

        return response["data"]["task_id"]

    def get_result(
        self, task_id: str, return_url: bool = True
    ) -> dict:
        """
        查询任务结果

        Args:
            task_id: 任务ID
            return_url: 是否返回图片URL，默认True

        Returns:
            dict包含:
                - status: 任务状态 (in_queue/generating/done/not_found/expired)
                - image_urls: 图片URL列表
                - binary_data_base64: 图片base64列表
        """
        body = {
            "req_key": "jimeng_t2i_v40",
            "task_id": task_id,
        }

        if return_url:
            body["req_json"] = json.dumps({"return_url": True})

        response = self._sign_request("POST", "CVSync2AsyncGetResult", body)

        if response.get("code") != 10000:
            raise JimengAIError(
                response.get("code", -1), response.get("message", "Unknown error")
            )

        return response.get("data", {})

    def generate_image(
        self,
        prompt: str,
        width: int = 2048,
        height: int = 2048,
        scale: float = 0.5,
        force_single: bool = True,
        timeout: int = 120,
        poll_interval: int = 2,
        image_urls: Optional[list[str]] = None,
    ) -> list[str]:
        """
        生成图片（同步接口，轮询等待结果）

        Args:
            prompt: 文本提示词
            width: 图片宽度，默认2048
            height: 图片高度，默认2048
            scale: 文本影响程度0-1，默认0.5
            force_single: 是否强制生成单图，默认True
            timeout: 超时时间（秒），默认120
            poll_interval: 轮询间隔（秒），默认2
            image_urls: 参考图片URL列表，可选

        Returns:
            图片URL列表

        Raises:
            JimengAIError: 生成失败时抛出
            TimeoutError: 超时时抛出
        """
        # 1. 提交任务
        task_id = self.submit_task(
            prompt=prompt,
            width=width,
            height=height,
            scale=scale,
            force_single=force_single,
            image_urls=image_urls,
        )

        # 2. 轮询查询结果
        start_time = time.time()
        while time.time() - start_time < timeout:
            result = self.get_result(task_id)

            status = result.get("status")

            if status == "done":
                image_urls = result.get("image_urls") or []
                if not image_urls:
                    # 尝试返回base64
                    binary_data = result.get("binary_data_base64") or []
                    if binary_data:
                        # 返回base64列表（需要调用方自行处理）
                        return binary_data
                    raise JimengAIError(-1, "No image returned")
                return image_urls

            elif status in ("not_found", "expired"):
                raise JimengAIError(-1, f"Task {status}")

            # 继续等待
            time.sleep(poll_interval)

        raise TimeoutError(f"Generate image timeout after {timeout}s")


def create_jimeng_client() -> JimengClient:
    """
    工厂函数：从环境变量创建JimengClient

    Returns:
        JimengClient实例
    """
    ak = os.getenv("JIMENG_ACCESS_KEY_ID")
    sk = os.getenv("JIMENG_SECRET_ACCESS_KEY")

    if not ak or not sk:
        raise ValueError(
            "JIMENG_ACCESS_KEY_ID and JIMENG_SECRET_ACCESS_KEY must be set"
        )

    return JimengClient(ak, sk)