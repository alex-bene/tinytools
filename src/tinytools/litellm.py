"""LiteLLM wrapper."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Union

import litellm
from dotenv import load_dotenv
from litellm import JSONSchemaValidationError, completion
from litellm.litellm_core_utils.json_validation_rule import validate_schema
from PIL.Image import Image
from tqdm import tqdm

from .logger import get_logger

if TYPE_CHECKING:
    from pydantic import BaseModel

load_dotenv()

ImageType = Union[str, Path, Image]

# Setup logger
logger = get_logger(__name__)
## Suppress litellm logs
litellm_logger = logging.getLogger("LiteLLM")
litellm_logger.setLevel(logging.WARNING)
litellm.suppress_debug_info = True


class LiteLLMModel:
    """LiteLLM wrapper.

    Args:
        model (str, optional): The name of the model to use. Defaults to "gemini/gemini-2.5".
        cache_folder (str | Path | None, optional): The path to the cache folder. Defaults to None.
        max_retries (int, optional): The maximum number of retries. Defaults to 3.
        ignore_not_found (bool, optional): Whether to ignore not found errors. Defaults to False.

    """

    def __init__(
        self,
        model: str = "gemini/gemini-2.5-pro",
        cache_folder: str | Path | None = None,
        max_retries: int = 3,
        ignore_not_found: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.max_retries = max_retries
        self.ignore_not_found = ignore_not_found

        # Setup cache folder
        self.cache_folder = (
            Path(cache_folder) if cache_folder is not None else Path.home() / ".cache" / "tinytools" / "vlm_cache"
        )
        self.cache_folder /= model.lower()
        self.cache_folder.mkdir(exist_ok=True, parents=True)

    @staticmethod
    def encode_image(image: ImageType) -> str:
        """Encode the image as a base64 string."""
        if isinstance(image, Image):
            # Convert PIL image to bytes
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

        with Path(image).open("rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    @staticmethod
    def completion_with_retries(
        model: str, messages: list[dict[str, str]], max_retries: int = 3, response_format: BaseModel | None = None
    ) -> dict[str, str]:
        """Completion with retries.

        This function wraps the default litellm completion function and retries "max_retries" times with a cooldown of
        70 seconds, if it fails for any reason.

        If the completion fails after all retries, it checks if the error is due to JSONSchemaValidationError and if so,
        it logs a warning message and returns an empty string. If the error is not due to JSONSchemaValidationError,
        it raises a RuntimeError.

        Args:
            model (str): The name of the model to use.
            messages (list[dict[str, str]]): The messages to send to the model.
            max_retries (int, optional): The maximum number of retries. Defaults to 3.
            response_format (BaseModel | None, optional): The response format. Defaults to None.

        Returns:
            dict[str, str]: The response from the model.

        Raises:
            RuntimeError: If the completion fails after all retries.

        """
        exception = None
        for _ in range(max_retries):
            try:
                response = completion(model=model, messages=messages)
                response_content = response.choices[0].message.content

                if response_content is None:
                    msg = "Failed to generate response. Maybe check your prompt."
                    raise ValueError(msg)  # noqa: TRY301

                if response_format is not None:
                    response_content = json.loads(response_content)
                    if isinstance(response_content, list):
                        response_content = response_content[0]
                    # gemini 2.5 pro exp does not follow the schema very well. Returning the whole response inside the
                    # properties field is quite common so we take this into account here instead of just retrying
                    response_content = response_content.get("properties", response_content)
                    validate_schema(schema=response_format.model_json_schema(), response=json.dumps(response_content))
                    # validation passes if there are extra keys in reponse_content, so remove them
                    response_format_fields = response_format.model_fields
                    response_content_keys = list(response_content.keys())
                    for field in response_content_keys:
                        if field not in response_format_fields:
                            response_content.pop(field)
            except Exception as e:  # noqa: BLE001, PERF203
                exception = e
                if isinstance(e, litellm.RateLimitError):
                    time.sleep(70)
            else:
                response.choices[0].message.content = (
                    response_content if response_format is None else json.dumps(response_content)
                )
                return response

        if isinstance(exception, JSONSchemaValidationError):
            msg = "Failed to validate the response. Retry later."
            logger.warning(msg)
            logger.warning(response_content)
            response.choices[0].message.content = ""
            return response
        if (
            isinstance(exception, ValueError)
            and exception.args[0] == "Failed to generate response. Maybe check your prompt."
        ):
            logger.warning("Failed to generate response. Maybe check your prompt.")
            response.choices[0].message.content = ""
            return response

        msg = f"Error encountered after {max_retries} retries."
        raise RuntimeError(msg) from exception

    def single_forward(
        self,
        prompt: str,
        system_prompt: str | None = None,
        images: list[ImageType] | ImageType | None = None,
        response_format: BaseModel | None = None,
    ) -> str:
        """Make a single forward pass through the VLM/LLM with an optional image, system prompt and response format.

        Args:
            prompt (str): The prompt to send to the model.
            system_prompt (str | None, optional): The system prompt to send to the model. Defaults to None.
            images (list[ImageType] | ImageType | None, optional): The images to send to the model.
                Defaults to None.
            response_format (pydantic.BaseModel | None, optional): The response format. Defaults to None.

        Returns:
            str: The response from the model.

        """
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
        if savepath.exists():
            with savepath.open("r") as fp:
                outputs = json.load(fp)
                if response_format is not None:
                    validate_schema(
                        schema=response_format.model_json_schema(), response=json.dumps(outputs["final_response"])
                    )
                return outputs
        elif self.ignore_not_found:
            return ""

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
        response = self.completion_with_retries(
            model=self.model, messages=messages, max_retries=self.max_retries, response_format=response_format
        )
        ## Check if the model did not finish as expected
        if response.choices[0].finish_reason != "stop":
            msg = f"The model did not finish as expected. Got finish_reason = {response.choices[0].finish_reason}"
            logger.warning(msg)
            return ""
        ## Append the response to the list of responses
        response_content = (
            response.choices[0].message.content
            if response_format is None
            else json.loads(response.choices[0].message.content)
        )

        # Save the output to cache if it finished as expected
        if response_content not in ("", None):
            with savepath.open("w") as fp:
                json.dump(response_content, fp)

        return response_content

    def forward(
        self,
        prompts: list[str],
        system_prompts: list[str | None] | str | None = None,
        images: list[list[ImageType] | ImageType | None] | None = None,
        response_format: BaseModel | None = None,
    ) -> list[str]:
        """Make a forward pass through the VLM/LLM with optional images, system prompts and response format.

        Args:
            prompts (list[str]): The prompts to send to the model.
            system_prompts (list[str | None] | str | None, optional): The system prompts to send to the model.
                Defaults to None.
            images (list[list[ImageType] | ImageType | None] | None, optional): The images to send to the model.
                Defaults to None.
            response_format (pydantic.BaseModel | None, optional): The response format. Defaults to None.

        Returns:
            list[str]: The responses from the model.

        """
        if not prompts:
            return []

        future_to_index = {}
        with ThreadPoolExecutor(max_workers=min(os.cpu_count(), len(prompts))) as executor:
            for idx, prompt in enumerate(prompts):
                future = executor.submit(
                    self.single_forward,
                    prompt=prompt,
                    system_prompt=system_prompts[idx] if isinstance(system_prompts, list) else system_prompts,
                    images=images[idx] if isinstance(images, list) else images,
                    response_format=response_format,
                )
                future_to_index[future] = idx

            results = [None] * len(prompts)
            for future in tqdm(as_completed(future_to_index), total=len(prompts), desc="Processing prompts"):
                results[future_to_index[future]] = future.result()

        return results
