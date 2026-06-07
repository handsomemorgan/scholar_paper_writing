"""
LLM客户端 - 统一的LLM API调用接口

支持 Anthropic Claude 和 OpenAI 兼容接口。
"""

import os
import logging
from typing import Optional, Dict, Any

import yaml
from anthropic import Anthropic
from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    """统一的LLM客户端，支持多种后端"""

    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        llm_config = config["llm"]
        self.provider = llm_config["provider"]
        self.model = llm_config["model"]
        self.max_tokens = llm_config.get("max_tokens", 4096)
        self.temperature = llm_config.get("temperature", 0.7)

        # 初始化对应的客户端
        if self.provider == "anthropic":
            api_key = os.getenv(llm_config.get("api_key_env", "ANTHROPIC_API_KEY"))
            if not api_key:
                raise ValueError(f"Environment variable {llm_config['api_key_env']} not set")
            self.anthropic_client = Anthropic(api_key=api_key)
            self.openai_client = None
        elif self.provider == "openai":
            api_key = os.getenv(llm_config.get("api_key_env", "OPENAI_API_KEY"))
            if not api_key:
                raise ValueError(f"Environment variable {llm_config['api_key_env']} not set")
            self.openai_client = OpenAI(api_key=api_key)
            self.anthropic_client = None
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        发送对话请求，返回文本响应。

        Args:
            system_prompt: 系统提示词
            user_message: 用户消息
            temperature: 温度参数(可选，覆盖配置)
            max_tokens: 最大token数(可选，覆盖配置)

        Returns:
            LLM的文本响应
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        try:
            if self.provider == "anthropic":
                return self._chat_anthropic(system_prompt, user_message, temp, max_tok)
            else:
                return self._chat_openai(system_prompt, user_message, temp, max_tok)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def _chat_anthropic(
        self, system_prompt: str, user_message: str, temperature: float, max_tokens: int
    ) -> str:
        """Anthropic Claude API 调用"""
        message = self.anthropic_client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return message.content[0].text

    def _chat_openai(
        self, system_prompt: str, user_message: str, temperature: float, max_tokens: int
    ) -> str:
        """OpenAI 兼容 API 调用"""
        response = self.openai_client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content

    def chat_with_json_output(
        self,
        system_prompt: str,
        user_message: str,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        发送对话请求，返回JSON格式响应。

        注意：当前实现通过prompt控制输出JSON格式，
        Anthropic/OpenAI原生JSON mode可在此扩展。
        """
        json_instruction = (
            "\n\nIMPORTANT: You MUST respond with valid JSON only. "
            "No markdown, no code fences, no extra text. Just pure JSON."
        )
        system_prompt += json_instruction

        response_text = self.chat(system_prompt, user_message, temperature=temperature)

        # 清理可能的 markdown 代码块标记
        response_text = response_text.strip()
        if response_text.startswith("```"):
            # 移除 ```json 和结尾的 ```
            response_text = response_text.split("\n", 1)[-1]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

        import json
        return json.loads(response_text.strip())
