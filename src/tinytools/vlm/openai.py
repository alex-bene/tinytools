"""OpenAI API wrapper."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL.Image import Image
from tqdm.asyncio import tqdm_asyncio

from tinytools.logger import get_logger
from tinytools.suppressors import suppress_logging

try:
    import openai
    import torch
    from dotenv import load_dotenv
    from jsonschema import validate
    from pydantic import ValidationError
except ImportError as e:
    msg = 'LLM features are not available. Please install the required dependencies with: pip install "tinytools[llm]"'
    raise ImportError(msg) from e

if TYPE_CHECKING:
    from pydantic import BaseModel
else:
    BaseModel = Any

load_dotenv()

ImageType = str | Path | Image

# Setup logger
logger = get_logger(__name__)
## Supress 'httpx' logs
logging.getLogger("httpx").setLevel(logging.WARNING)


class OpenAIAPIModel:
    """LiteLLM wrapper."""

    def __init__(
        self,
        model: str = "gemini/gemini-2.5-pro",
        cache_folder: str | Path | None = None,
        max_retries: int = openai.DEFAULT_MAX_RETRIES,
        ignore_not_found: bool = False,
        ignore_cache: bool = False,
        no_cache: bool = False,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "",
        timeout: int = 600,
        project: str | None = None,
        organization: str | None = None,
        vllm_gpu_id: int = 0,
        vllm_port: int = 8000,
    ) -> None:
        """Initialize the LiteLLM model.

        Args:
            model (str, optional): The name of the model to use. Defaults to "gemini/gemini-2.5".
            cache_folder (str | Path | None, optional): The path to the cache folder. Defaults to None.
            max_retries (int, optional): The maximum number of retries. Defaults to 2.
            ignore_not_found (bool, optional): Whether to ignore requests not in cache. Defaults to False.
            ignore_cache (bool, optional): Whether to ignore cache hits. Defaults to False.
            no_cache (bool, optional): Whether to disable caching. Defaults to False.
            base_url (str, optional): The base URL of the API. Defaults to "http://localhost:8000/v1".
            api_key (str, optional): The API key to use. Defaults to "".
            timeout (int, optional): The timeout for requests in seconds. Defaults to 600.
            project (str | None, optional): The project to use. Defaults to None.
            organization (str | None, optional): The organization to use. Defaults to None.
            vllm_gpu_id (int, optional): The GPU ID to use. Defaults to 0.
            vllm_port (int, optional): The port to use for the vLLM engine. Defaults to 8000.

        """
        super().__init__()
        self.model = model
        self.max_retries = max_retries
        self.ignore_not_found = ignore_not_found
        self.ignore_cache = ignore_cache
        self.no_cache = no_cache
        self.vllm_gpu_id = vllm_gpu_id
        self.vllm_port = vllm_port
        self.vllm_engine_process = None

        # Use the async client from the openai library
        self.client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            project=project,
            organization=organization,
        )
        # Setup cache folder
        self.cache_folder = (
            Path(cache_folder) if cache_folder is not None else Path.home() / ".cache" / "tinytools" / "vlm_cache"
        )
        self.cache_folder /= model.lower()
        self.cache_folder.mkdir(exist_ok=True, parents=True)

    def vllm_engine_running(self) -> bool:
        """Check if the vLLM engine is running."""
        try:
            self.forward(["test"], no_progress_bar=True, no_cache=True, ignore_cache=True)
        except openai.APIConnectionError:
            return False
        else:
            return True

    def vllm_engine_start(self, synchronous: bool = False) -> subprocess.Popen | None:
        """Start the VLLM engine."""
        with suppress_logging():
            is_running = self.vllm_engine_running()
        if is_running:
            logger.info("vLLM engine already running")
            return None

        # Basic inputs validation
        if not isinstance(self.model, str) or not self.model.isprintable():
            msg = "Invalid model name"
            raise ValueError(msg)
        if not isinstance(self.vllm_port, int) or not (0 < self.vllm_port < 65536):
            msg = "Invalid port number"
            raise ValueError(msg)
        if not isinstance(self.vllm_gpu_id, int) or torch.cuda.device_count() < self.vllm_gpu_id < 0:
            msg = "Invalid GPU ID"
            raise ValueError(msg)

        self.vllm_engine_process = subprocess.Popen(  # noqa: S603
            [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                self.model,
                "--headless",
                "--dtype",
                "auto",
                "--limit-mm-per-prompt",
                '{"image":1,"video":0}',
                "--port",
                str(self.vllm_port),
            ],
            env={"CUDA_VISIBLE_DEVICES": str(self.vllm_gpu_id)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            start_new_session=True,
        )
        if synchronous:
            with suppress_logging():
                while not self.vllm_engine_running() and self.vllm_engine_process.poll() is None:
                    time.sleep(1)
        return self.vllm_engine_process

    def vllm_engine_stop(self) -> None:
        """Stop the vLLM engine gracefully."""
        if not self.vllm_engine_running():
            logger.info("vLLM engine not running")
            return

        if self.vllm_engine_process:
            try:
                # Terminate the process
                self.vllm_engine_process.terminate()
                self.vllm_engine_process.wait(timeout=10)
                logger.info("Stopped vLLM engine (PID: %d)", self.vllm_engine_process.pid)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate gracefully
                self.vllm_engine_process.kill()
                logger.info("Force killed vLLM engine (PID: %d)", self.vllm_engine_process.pid)
            except Exception:
                logger.exception("Error stopping vLLM engine")
            finally:
                self.vllm_engine_process = None
        else:
            logger.info("vLLM engine was not started from this process")

    @staticmethod
    def encode_image(image: ImageType) -> str:
        """Encode the image as a base64 string."""
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        # Convert PIL image to bytes
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    async def completion_with_retries(
        self, messages: list[dict[str, str]], response_format: type[BaseModel] | None = None, max_tokens: int = 8192
    ) -> dict[str, str] | None:
        """Completion with retries.

        This function wraps the default litellm completion function and retries "max_retries" times with a cooldown of
        70 seconds, if it fails for any reason.

        If the completion fails after all retries, it checks if the error is due to JSONSchemaValidationError and if so,
        it logs a warning message and returns an empty string. If the error is not due to JSONSchemaValidationError,
        it raises a RuntimeError.

        Args:
            model (str): The name of the model to use.
            messages (list[dict[str, str]]): The messages to send to the model.
            response_format (BaseModel | None, optional): The response format. Defaults to None.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 8192.

        Returns:
            dict[str, str] | None: The response from the model or None if the completion fails after all retries.

        Raises:
            RuntimeError: If the completion fails after all retries.

        """
        exception = None
        for _ in range(self.max_retries + 1):
            try:
                response = await self.client.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    response_format=response_format if response_format is not None else openai.omit,
                    max_tokens=max_tokens,
                )
                response_content: str = response.choices[0].message.content

                if response_content is None:
                    msg = "Failed to generate response. Maybe check your prompt."
                    raise ValueError(msg)  # noqa: TRY301
            except Exception as e:  # noqa: BLE001, PERF203
                exception = e
            else:
                return response

        if isinstance(exception, ValidationError):
            msg = "Failed to validate the response."
            logger.warning(msg)
            return None
        if isinstance(exception, openai.LengthFinishReasonError):
            msg = "Max tokens reached. Increase max_tokens."
            logger.warning(msg)
            return None
        if (
            isinstance(exception, ValueError)
            and exception.args[0] == "Failed to generate response. Maybe check your prompt."
        ):
            logger.warning("Failed to generate response. Maybe check your prompt.")
            return None

        exception.args = (
            f"Error encountered after {self.max_retries} retries Original error: {exception.args[0]}",
            *exception.args[1:],
        )
        raise exception

    async def single_forward(
        self,
        prompt: str,
        system_prompt: str | None = None,
        images: list[ImageType] | ImageType | None = None,
        response_format: BaseModel | None = None,
        no_cache: bool | None = None,
        ignore_cache: bool | None = None,
        max_tokens: int = 8192,
    ) -> str:
        """Make a single forward pass through the VLM/LLM with an optional image, system prompt and response format.

        Args:
            prompt (str): The prompt to send to the model.
            system_prompt (str | None, optional): The system prompt to send to the model. Defaults to None.
            images (list[ImageType] | ImageType | None, optional): The images to send to the model.
                Defaults to None.
            response_format (pydantic.BaseModel | None, optional): The response format. Defaults to None.
            no_cache (bool | None, optional): Whether to disable caching. Defaults to False.
            ignore_cache (bool | None, optional): Whether to ignore the cache. Defaults to False.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 8192.

        Returns:
            str | None: The response from the model.

        """
        no_cache = self.no_cache if no_cache is None else no_cache
        ignore_cache = self.ignore_cache if ignore_cache is None else ignore_cache

        if system_prompt == "":
            system_prompt = None

        # Encode image
        if isinstance(images, (str, Path, Image)):
            images = [images]
        base64_images = [self.encode_image(image) for image in images] if images is not None else []

        # Create unique savepath
        hashkey = hashlib.sha256(
            str(
                [
                    system_prompt,
                    prompt,
                    base64_images,
                    response_format.model_json_schema() if response_format is not None else None,
                ]
            ).encode("utf-8")
        ).hexdigest()
        savepath = self.cache_folder / f"{hashkey}.json"
        # Alt savepaths (we use exp and preview models interchangeably)
        savepath_alt = Path(savepath.as_posix().replace("exp", "preview"))
        savepath_alt = Path(savepath.as_posix().replace("preview", "exp")) if savepath_alt == savepath else savepath_alt
        savepath = savepath_alt if savepath_alt.exists() and not savepath.exists() else savepath

        # Check if the answers file exists
        if savepath.exists() and not ignore_cache:
            with savepath.open("r") as fp:
                outputs = json.load(fp)
                if response_format is not None:
                    validate(schema=response_format.model_json_schema(), response=outputs)
                return outputs
        elif self.ignore_not_found:
            return None

        # Generate the response
        messages = [{"role": "system", "content": system_prompt}] if system_prompt is not None else []
        ## Add current prompt to the message history
        content = (
            prompt
            if not base64_images
            else [{"type": "text", "text": prompt}]
            + [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                for base64_image in base64_images
            ]
        )
        messages.append({"role": "user", "content": content})
        ## Generate reply and apply response_format
        response = await self.completion_with_retries(
            messages=messages,
            response_format=response_format if response_format is not None else openai.omit,
            max_tokens=max_tokens,
        )  # returns None if failed

        ## Check if the model did not finish as expected
        if isinstance(response, str | None):
            return None
        if response.choices[0].finish_reason != "stop":
            msg = f"The model did not finish as expected. Got finish_reason = {response.choices[0].finish_reason}"
            logger.warning(msg)
            return None
        ## Append the response to the list of responses
        response_content = (
            response.choices[0].message.content
            if response_format is None or response.choices[0].message.content is None
            else json.loads(response.choices[0].message.content)
        )

        # Save the output to cache if it finished as expected
        if response_content is not None and not no_cache:
            savepath.parent.mkdir(parents=True, exist_ok=True)
            with savepath.open("w") as fp:
                json.dump(response_content, fp)

        return response_content

    async def async_forward(
        self,
        prompts: list[str],
        system_prompts: list[str | None] | str | None = None,
        images: list[list[ImageType] | ImageType | None] | None = None,
        response_format: BaseModel | None = None,
        no_progress_bar: bool = False,
        no_cache: bool = False,
        ignore_cache: bool | None = None,
        max_tokens: int = 8192,
    ) -> list[str | None]:
        """Make a forward pass through the VLM/LLM with optional images, system prompts and response format.

        Args:
            prompts (list[str]): The prompts to send to the model.
            system_prompts (list[str | None] | str | None, optional): The system prompts to send to the model.
                Defaults to None.
            images (list[list[ImageType] | ImageType | None] | None, optional): The images to send to the model.
                Defaults to None.
            response_format (pydantic.BaseModel | None, optional): The response format. Defaults to None.
            no_progress_bar (bool, optional): Whether to disable the progress bar. Defaults to False.
            no_cache (bool | None, optional): Whether to disable caching. Defaults to False.
            ignore_cache (bool | None, optional): Whether to ignore the cache. Defaults to None.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 8192.

        Returns:
            list[str | None]: The responses from the model.

        """
        if not prompts:
            return []

        tasks = [
            self.single_forward(
                prompt=prompt,
                system_prompt=system_prompts[idx] if isinstance(system_prompts, list) else system_prompts,
                images=images[idx] if isinstance(images, list) else images,
                response_format=response_format,
                no_cache=no_cache,
                ignore_cache=ignore_cache,
                max_tokens=max_tokens,
            )
            for idx, prompt in enumerate(prompts)
        ]

        return await tqdm_asyncio.gather(*tasks, desc="vLLM inference", total=len(prompts), disable=no_progress_bar)

    def forward(
        self,
        prompts: list[str],
        system_prompts: list[str | None] | str | None = None,
        images: list[list[ImageType] | ImageType | None] | None = None,
        response_format: BaseModel | None = None,
        no_progress_bar: bool = False,
        no_cache: bool | None = None,
        ignore_cache: bool | None = None,
        max_tokens: int = 8192,
    ) -> list[str | None]:
        """Make a forward pass through the VLM/LLM with optional images, system prompts and response format.

        Args:
            prompts (list[str]): The prompts to send to the model.
            system_prompts (list[str | None] | str | None, optional): The system prompts to send to the model.
                Defaults to None.
            images (list[list[ImageType] | ImageType | None] | None, optional): The images to send to the model.
                Defaults to None.
            response_format (pydantic.BaseModel | None, optional): The response format. Defaults to None.
            no_progress_bar (bool, optional): Whether to disable the progress bar. Defaults to False.
            no_cache (bool | None, optional): Whether to disable caching. Defaults to False.
            ignore_cache (bool | None, optional): Whether to ignore the cache. Defaults to None.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 8192.

        Returns:
            list[str | None]: The responses from the model.

        """
        # check if I am alread in an event loop
        async_run = self.async_forward(
            prompts=prompts,
            system_prompts=system_prompts,
            images=images,
            response_format=response_format,
            no_progress_bar=no_progress_bar,
            no_cache=no_cache,
            ignore_cache=ignore_cache,
            max_tokens=max_tokens,
        )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(async_run)
        else:
            if not loop.is_running():
                return asyncio.run_coroutine_threadsafe(async_run, loop).result()

            # we are inside an event loop → run in a new thread
            with ThreadPoolExecutor() as executor:
                # Submit the async function to run in a separate thread
                future = executor.submit(asyncio.run, async_run)
                return future.result()  # Blocks until completion
