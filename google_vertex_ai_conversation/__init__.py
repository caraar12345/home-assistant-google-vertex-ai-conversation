"""The Google Vertex AI Conversation integration."""

from __future__ import annotations

from functools import partial
import json
import logging
import mimetypes
from pathlib import Path
from typing import Literal

from google.api_core.exceptions import ClientError
import google.oauth2.service_account as google_service_account
import vertexai
import vertexai.preview.generative_models as vertexai_genmodels
import voluptuous as vol

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import MATCH_ALL
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.exceptions import (
    ConfigEntryNotReady,
    HomeAssistantError,
    TemplateError,
)
from homeassistant.helpers import config_validation as cv, intent, template
from homeassistant.helpers.typing import ConfigType
from homeassistant.util import ulid

from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROJECT_ID,
    CONF_PROMPT,
    CONF_REGION,
    CONF_SERVICE_ACCOUNT_JSON,
    CONF_SERVICE_MODEL,
    CONF_TEMPERATURE,
    CONF_TOP_K,
    CONF_TOP_P,
    DEFAULT_CHAT_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_SERVICE_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_VISION_MODEL,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)
SERVICE_GENERATE_CONTENT = "generate_content"
CONF_IMAGE_FILENAME = "image_filename"

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up Google Generative AI Conversation."""

    async def generate_content(call: ServiceCall) -> ServiceResponse:
        """Generate content from text and optionally images."""
        prompt_parts = [call.data[CONF_PROMPT]]
        image_filenames = call.data[CONF_IMAGE_FILENAME]
        model_name = (
            call.data.get(CONF_SERVICE_MODEL, DEFAULT_SERVICE_MODEL)
            if not image_filenames
            else call.data.get(CONF_SERVICE_MODEL, DEFAULT_VISION_MODEL)
        )

        for image_filename in image_filenames:
            if not hass.config.is_allowed_path(image_filename):
                raise HomeAssistantError(
                    f"Cannot read `{image_filename}`, no access to path; "
                    "`allowlist_external_dirs` may need to be adjusted in "
                    "`configuration.yaml`"
                )
            if not Path(image_filename).exists():
                raise HomeAssistantError(f"`{image_filename}` does not exist")
            mime_type, _ = mimetypes.guess_type(image_filename)
            if mime_type is None or not mime_type.startswith("image"):
                raise HomeAssistantError(f"`{image_filename}` is not an image")
            prompt_parts.append(
                vertexai_genmodels.Image.from_bytes(
                    await hass.async_add_executor_job(Path(image_filename).read_bytes)
                )
            )

        model = vertexai_genmodels.GenerativeModel(model_name=model_name)

        try:
            response = await model.generate_content_async(prompt_parts)
        except (
            ClientError,
            ValueError,
            vertexai_genmodels.ResponseValidationError,
        ) as err:
            raise HomeAssistantError(f"Error generating content: {err}") from err

        return {"text": response.text}

    hass.services.async_register(
        DOMAIN,
        SERVICE_GENERATE_CONTENT,
        generate_content,
        schema=vol.Schema(
            {
                vol.Required(CONF_PROMPT): cv.string,
                vol.Optional(CONF_IMAGE_FILENAME, default=[]): vol.All(
                    cv.ensure_list, [cv.string]
                ),
                vol.Optional(CONF_SERVICE_MODEL): cv.string,
            }
        ),
        supports_response=SupportsResponse.ONLY,
    )
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Google Generative AI Conversation from a config entry."""
    creds = google_service_account.Credentials.from_service_account_info(
        json.loads(entry.data[CONF_SERVICE_ACCOUNT_JSON])
    ).with_quota_project(entry.data[CONF_PROJECT_ID])

    vertexai.init(
        project=entry.data[CONF_PROJECT_ID],
        location=entry.data[CONF_REGION],
        credentials=creds,
    )

    try:
        await hass.async_add_executor_job(
            partial(
                vertexai_genmodels.GenerativeModel,
                entry.options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL),
            )
        )
    except ClientError as err:
        if err.reason == "API_KEY_INVALID":
            _LOGGER.error("Invalid API key: %s", err)
            return False
        raise ConfigEntryNotReady(err) from err

    conversation.async_set_agent(hass, entry, GoogleGenerativeAIAgent(hass, entry))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload GoogleGenerativeAI."""
    vertexai.init(credentials=None)
    conversation.async_unset_agent(hass, entry)
    return True


class GoogleGenerativeAIAgent(conversation.AbstractConversationAgent):
    """Google Generative AI conversation agent."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.hass = hass
        self.entry = entry
        self.history: dict[str, list[vertexai_genmodels.Part]] = {}

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process a sentence."""
        raw_prompt = self.entry.options.get(CONF_PROMPT, DEFAULT_PROMPT)
        model = vertexai_genmodels.GenerativeModel(
            model_name=self.entry.options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL),
            generation_config={
                "temperature": self.entry.options.get(
                    CONF_TEMPERATURE, DEFAULT_TEMPERATURE
                ),
                "top_p": self.entry.options.get(CONF_TOP_P, DEFAULT_TOP_P),
                "top_k": self.entry.options.get(CONF_TOP_K, DEFAULT_TOP_K),
                "max_output_tokens": self.entry.options.get(
                    CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS
                ),
            },
        )
        _LOGGER.debug("Model: %s", model)

        if user_input.conversation_id in self.history:
            conversation_id = user_input.conversation_id
            messages = self.history[conversation_id]
        else:
            conversation_id = ulid.ulid_now()
            messages = [{}, {}]

        try:
            prompt = self._async_generate_prompt(raw_prompt)
        except TemplateError as err:
            _LOGGER.error("Error rendering prompt: %s", err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Sorry, I had a problem with my template: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        messages[0] = vertexai_genmodels.Content(
            role="user", parts=[vertexai_genmodels.Part.from_text(prompt)]
        )
        messages[1] = vertexai_genmodels.Content(
            role="model", parts=[vertexai_genmodels.Part.from_text("Ok")]
        )

        _LOGGER.debug("Input: '%s' with history: %s", user_input.text, messages)

        chat = model.start_chat(history=messages)
        try:
            chat_response = await chat.send_message_async(user_input.text)
        except (
            ClientError,
            ValueError,
            vertexai_genmodels.ResponseValidationError,
        ) as err:
            _LOGGER.error("Error sending message: %s", err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Sorry, I had a problem talking to Google Generative AI: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        _LOGGER.debug("Response: %s", chat_response.to_dict)
        self.history[conversation_id] = chat.history

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(chat_response.text)
        return conversation.ConversationResult(
            response=intent_response, conversation_id=conversation_id
        )

    def _async_generate_prompt(self, raw_prompt: str) -> str:
        """Generate a prompt for the user."""
        return template.Template(raw_prompt, self.hass).async_render(
            {
                "ha_name": self.hass.config.location_name,
            },
            parse_result=False,
        )
