"""VLLM wrapper."""

from __future__ import annotations

import base64
import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL.Image import Image

from tinytools.logger import get_logger

try:
    from dotenv import load_dotenv
    from jsonschema import ValidationError, validate
    from vllm import LLM, RequestOutput, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
except ImportError as e:
    msg = 'LLM features are not available. Please install the required dependencies with: pip install "tinytools[llm]"'
    raise ImportError(msg) from e

if TYPE_CHECKING:
    from pydantic import BaseModel

load_dotenv()

ImageType = str | Path | Image


logger = get_logger(__name__)


class FinishReasonError(Exception):
    """Exception raised when the model did not finish as expected."""


class VLLMModel:
    """LiteLLM wrapper."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        cache_folder: str | Path | None = None,
        max_retries: int = 3,
        ignore_not_found: bool = False,
        ignore_errors: bool = True,
        gpu_memory_utilization: float = 0.9,
        vllm_model_kwargs: dict[str, Any] | None = None,
        ignore_cache: bool = False,
        no_cache: bool = False,
    ) -> None:
        """Initialize the VLLM model.

        Args:
            model_name (str, optional): The name of the model to use. Defaults to "Qwen/Qwen2.5-VL-7B-Instruct".
            cache_folder (str | Path | None, optional): The path to the cache folder. If None, it will be set to
                "~/.cache/tinytools/vlm_cache". Defaults to None.
            max_retries (int, optional): The maximum number of retries. Defaults to 3.
            ignore_not_found (bool, optional): Whether to ignore not found errors. Defaults to False.
            ignore_errors (bool, optional): Whether to ignore `ValidationError`s and `FinishReasonError`s in the model
                completion. Defaults to True.
            gpu_memory_utilization (float, optional): The max GPU memory utilization. Defaults to 0.9.
            vllm_model_kwargs (dict[str, Any] | None, optional): The kwargs to pass to the VLLM model. Defaults to None.
            ignore_cache (bool, optional): Whether to ignore cache hits. Defaults to False.
            no_cache (bool, optional): Whether to disable caching. Defaults to False.

        """
        super().__init__()
        self.max_retries = max_retries
        self.ignore_errors = ignore_errors
        self.ignore_not_found = ignore_not_found
        self.ignore_cache = ignore_cache
        self.no_cache = no_cache
        self.model = None
        if not ignore_not_found:
            self.model = LLM(
                model=model_name, gpu_memory_utilization=gpu_memory_utilization, **(vllm_model_kwargs or {})
            )

        # Setup cache folder
        self.cache_folder = (
            Path(cache_folder) if cache_folder is not None else Path.home() / ".cache" / "tinytools" / "vlm_cache"
        )
        self.cache_folder /= model_name.lower()
        self.cache_folder.mkdir(exist_ok=True, parents=True)

    def completion_with_retries(
        self,
        messages: list[dict[str, str]],
        hashkeys: list[str],
        sampling_params: SamplingParams,
        response_format: BaseModel | None = None,
        no_progress_bar: bool = False,
    ) -> list[str, dict[str, Any]]:
        """Completion with retries.

        This function wraps the default VLLM chat function and retries `self.max_retries` times if it fails due to
        schema ValidationError or dues to the generation not finishing gracefully (FinishReasonError).

        If the completion fails after all retries and `self.ignore_errors` is True, it returns an empty string.
        Otherwise, it raises the corresponding error.

        Args:
            model (str): The name of the model to use.
            messages (list[dict[str, str]]): The messages to send to the model.
            hashkeys (list[str]): The hashkeys of each message to use for keeping track of the successful completions.
            sampling_params (SamplingParams): The sampling parameters for the generation.
            response_format (BaseModel | None, optional): The response format. Defaults to None.
            no_progress_bar (bool, optional): Whether to hide the progress bar. Defaults to False.

        Returns:
            list[str, dict[str, Any]]: The response from the model.

        Raises:
            JSONDecodeError: If the completion fails due to JSONDecodeError (the output is not json) after all retries
                and `self.ignore_errors` is False.
            ValidationError: If the completion fails due to json schema ValidationError (the output is json but does not
                match the response_format) after all retries and `self.ignore_errors` is False.
            FinishReasonError: If the completion fails due to the generation not finishing gracefully after all retries
                and `self.ignore_errors` is False.

        """
        current_hashkeys = hashkeys.copy()
        current_messages = messages.copy()
        last_exception = None
        responses_dict = {}
        try_counter = 0
        while current_messages:
            if try_counter > max(self.max_retries, 0):
                last_exception.add_note(f"Error encountered after {try_counter - 1} retries.")
                if self.ignore_errors:
                    logger.exception(last_exception)
                    responses_dict |= dict.fromkeys(current_hashkeys, None)
                    break
                raise last_exception
            responses = self.model.chat(
                messages=current_messages, sampling_params=sampling_params, use_tqdm=not no_progress_bar
            )
            idxs_to_pop = set()
            for idx, response in enumerate(responses):
                try:
                    self.check_finish_reason(response)
                    if response_format is not None:
                        validate(json.loads(response.outputs[0].text), schema=response_format.model_json_schema())
                except (FinishReasonError, ValidationError, json.JSONDecodeError) as e:  # noqa: PERF203
                    last_exception = e
                else:
                    formatted_response = response.outputs[0].text
                    if response_format is not None:
                        formatted_response = json.loads(formatted_response)
                    responses_dict[current_hashkeys[idx]] = formatted_response
                    idxs_to_pop.add(idx)
            current_hashkeys = [hashkey for idx, hashkey in enumerate(current_hashkeys) if idx not in idxs_to_pop]
            current_messages = [message for idx, message in enumerate(current_messages) if idx not in idxs_to_pop]
            try_counter += 1

        return [responses_dict[hashkey] for hashkey in hashkeys]

    def check_finish_reason(self, response: RequestOutput) -> None:
        """Check if the response has finished generation gracefull, raise an error if not."""
        finish_reason = response.outputs[0].finish_reason
        if finish_reason != "stop":
            msg = f"The model did not finish as expected. Got finish_reason = {finish_reason}"
            raise FinishReasonError(msg)

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

    def prepare_message(
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

        # Check if the answers file exists
        if savepath.exists() and not self.ignore_cache:
            with savepath.open("r") as fp:
                outputs = json.load(fp)
                if response_format is not None:
                    validate(outputs, schema=response_format.model_json_schema())
                return None, outputs, savepath, hashkey
        elif self.ignore_not_found:
            return None, "", savepath, hashkey

        # Generate the response
        messages = [{"role": "system", "content": system_prompt}] if system_prompt is not None else []
        ## Add current prompt to the message history
        content = (
            prompt
            if not images
            else [{"type": "text", "text": prompt}]
            + [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                for base64_image in base64_images
            ]
        )
        messages.append({"role": "user", "content": content})
        return messages, None, savepath, hashkey

    def forward(
        self,
        prompts: list[str],
        system_prompts: list[str | None] | str | None = None,
        images: list[list[ImageType] | ImageType | None] | None = None,
        response_format: BaseModel | None = None,
        no_progress_bar: bool = False,
        max_tokens: int = 4096,
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
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 4096.

        Returns:
            list[str | None]: The responses from the model.

        """
        if not prompts:
            return []

        messages, outputs, savepaths, hashkeys = zip(
            *[
                self.prepare_message(
                    prompt=prompt,
                    system_prompt=system_prompts[idx] if isinstance(system_prompts, list) else system_prompts,
                    images=images[idx] if isinstance(images, list) else images,
                    response_format=response_format,
                )
                for idx, prompt in enumerate(prompts)
            ],
            strict=True,
        )
        outputs = list(outputs)  # None means cache miss, "" means ignore not found

        non_cached_idxs = [idx for idx, output in enumerate(outputs) if output is None]
        logger.info("Using %d cached responses.", len(messages) - len(non_cached_idxs))

        outputs = [output if output != "" else None for output in outputs]
        if not non_cached_idxs:
            return outputs

        messages = [message for idx, message in enumerate(messages) if idx in non_cached_idxs]
        hashkeys = [hashkey for idx, hashkey in enumerate(hashkeys) if idx in non_cached_idxs]

        ## Generate reply and apply response_format
        sampling_params = self.model.get_default_sampling_params()
        sampling_params.max_tokens = max_tokens
        if response_format is not None:
            guided_decoding_params = GuidedDecodingParams(json=response_format.model_json_schema())
            sampling_params.guided_decoding = guided_decoding_params

        # If response_format is not None, validate the response and retry `self.max_retries` times if validation fails
        responses = self.completion_with_retries(
            messages=messages,
            hashkeys=hashkeys,
            response_format=response_format,
            no_progress_bar=no_progress_bar,
            sampling_params=sampling_params,
        )  # returns None if failed

        # Add cached responses
        if non_cached_idxs:
            savepaths[non_cached_idxs[0]].parent.mkdir(parents=True, exist_ok=True)
        for non_cached_idx, response in zip(non_cached_idxs, responses, strict=True):
            # Append the response to the list of responses
            outputs[non_cached_idx] = response
            # Save the output to cache if it finished as expected
            if response is not None and not self.no_cache:
                with savepaths[non_cached_idx].open("w") as fp:
                    json.dump(response, fp)

        return outputs
