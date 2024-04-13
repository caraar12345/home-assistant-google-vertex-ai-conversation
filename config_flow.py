"""Config flow for Google Vertex AI Conversation integration."""

from __future__ import annotations

from functools import partial
import json
import logging
import types
from types import MappingProxyType
from typing import Any

from google.api_core.exceptions import ClientError
import google.oauth2.service_account as google_service_account
import vertexai
import vertexai.preview.generative_models
import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
)

from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_POSSIBLE_REGIONS,
    CONF_PROJECT_ID,
    CONF_PROMPT,
    CONF_REGION,
    CONF_SERVICE_ACCOUNT_JSON,
    CONF_TEMPERATURE,
    CONF_TOP_K,
    CONF_TOP_P,
    DEFAULT_CHAT_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_PROJECT_ID): str,
        vol.Required(CONF_REGION): SelectSelector(
            SelectSelectorConfig(
                options=CONF_POSSIBLE_REGIONS, mode=SelectSelectorMode.DROPDOWN
            ),
        ),
        vol.Required(CONF_SERVICE_ACCOUNT_JSON): str,
    }
)

DEFAULT_OPTIONS = types.MappingProxyType(
    {
        CONF_PROMPT: DEFAULT_PROMPT,
        CONF_CHAT_MODEL: DEFAULT_CHAT_MODEL,
        CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
        CONF_TOP_P: DEFAULT_TOP_P,
        CONF_TOP_K: DEFAULT_TOP_K,
        CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
    }
)


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """Validate the user input allows us to connect.

    Data has the keys from STEP_USER_DATA_SCHEMA with values provided by the user.
    """
    creds = google_service_account.Credentials.from_service_account_info(
        json.loads(data[CONF_SERVICE_ACCOUNT_JSON])
    ).with_quota_project(data[CONF_PROJECT_ID])
    # genai.configure(api_key=data[CONF_API_KEY])
    vertexai.init(
        project=data[CONF_PROJECT_ID], location=data[CONF_REGION], credentials=creds
    )
    model = vertexai.generative_models.GenerativeModel("gemini-pro")
    await hass.async_add_executor_job(partial(model.generate_content, "Hello, Gemini!"))


class GoogleVertexAIConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Google Generative AI Conversation."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user", data_schema=STEP_USER_DATA_SCHEMA
            )

        errors = {}

        try:
            await validate_input(self.hass, user_input)
        except ClientError as err:
            if err.reason == "API_KEY_INVALID":
                errors["base"] = "invalid_auth"
            else:
                errors["base"] = err.reason
        except Exception:  # pylint: disable=broad-except
            _LOGGER.exception("Unexpected exception")
            errors["base"] = "unknown"
        else:
            return self.async_create_entry(
                title="Google Vertex AI Conversation", data=user_input
            )

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )

    @staticmethod
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> OptionsFlow:
        """Create the options flow."""
        return GoogleVertexAIOptionsFlow(config_entry)


class GoogleVertexAIOptionsFlow(OptionsFlow):
    """Google Vertex AI config flow options handler."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(
                title="Google Vertex AI Conversation", data=user_input
            )
        schema = google_vertex_ai_config_option_schema(self.config_entry.options)
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
        )


def google_vertex_ai_config_option_schema(
    options: MappingProxyType[str, Any],
) -> dict:
    """Return a schema for Google Vertex AI completion options."""
    if not options:
        options = DEFAULT_OPTIONS
    return {
        vol.Optional(
            CONF_PROMPT,
            description={"suggested_value": options[CONF_PROMPT]},
            default=DEFAULT_PROMPT,
        ): TemplateSelector(),
        vol.Optional(
            CONF_CHAT_MODEL,
            description={
                "suggested_value": options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
            },
            default=DEFAULT_CHAT_MODEL,
        ): str,
        vol.Optional(
            CONF_TEMPERATURE,
            description={"suggested_value": options[CONF_TEMPERATURE]},
            default=DEFAULT_TEMPERATURE,
        ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
        vol.Optional(
            CONF_TOP_P,
            description={"suggested_value": options[CONF_TOP_P]},
            default=DEFAULT_TOP_P,
        ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
        vol.Optional(
            CONF_TOP_K,
            description={"suggested_value": options[CONF_TOP_K]},
            default=DEFAULT_TOP_K,
        ): int,
        vol.Optional(
            CONF_MAX_TOKENS,
            description={"suggested_value": options[CONF_MAX_TOKENS]},
            default=DEFAULT_MAX_TOKENS,
        ): int,
    }
