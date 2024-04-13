# Google Vertex AI Conversation

Basically just the [built-in integration](https://www.home-assistant.io/integrations/google_generative_ai_conversation/) but it uses the Vertex API rather than Gemini and therefore works in countries in which the Gemini API is not supported.

It will probably break. But it seems to work mostly!

**Huge thanks to [@tronikos](https://github.com/tronikos) for writing the vast majority of this integration!**

## Usage

- Create a service account in a GCP project with the `Vertex AI User` role.
- Generate a JSON key
- Paste that into the Service Account JSON box alongside the GCP project ID and the region you want to use.
