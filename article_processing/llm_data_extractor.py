from openai import OpenAI
import base64
import os
from config import Config


class LLMDataExtractor:
    def __init__(self, config: Config):
        """Initialize with config object."""
        self.config = config
        self.api_key = config.openai_api_key
        self.client = OpenAI(api_key=self.api_key)

    def _encode_image_to_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def create_message(
        self,
        input_text: str,
        image_path: str = None,
        message_type: str = "analysis",
        plot_num: int = None,
    ) -> list:
        """
        Create different message structures based on input type.

        Args:
            input_text: The text content to analyze
            message_type: Type of message to create ('analysis', 'extraction', 'summary', 'classification')

        Returns:
            List of message dictionaries for OpenAI API
        """

        if image_path:
            # Encode image to base64 if provided
            image_base64 = self._encode_image_to_base64(image_path)

        if message_type == "ThT_plot_identifier":
            messages = [
                {
                    "role": "system",
                    "content": """You are given a scientific figure where each plot is labeled with a unique number and enclosed within a bounding box.

        Your task is to identify which of these plots are Thioflavin T (ThT) fluorescence vs. time plots. Only consider plots that have sequentially labelled bounding boxes.

        The ThT time plots are line or scatter plots (not bar charts, images, or tables). They have a fluorescence y-axis and a time x-axis. The y-axis label may include "ThT", "Thioflavin T", or "fluorescence". The x-axis label may include "time", "minutes", or "seconds".
        Return ONLY a Python-style list of the relevant plot numbers. For example: [1, 3, 4]. If there are none, return an empty list: [].
        
        Do not include any additional explanation or text.""",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "auto",
                            },
                        },
                    ],
                },
            ]

        elif message_type == "variables_extractor" and plot_num:
            user_content = [
                {
                    "type": "text",
                    "text": input_text,  # Should include caption + surrounding context
                }
            ]

            # Add image if provided
            if image_path:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "auto",
                        },
                    }
                )

            messages = [
                {
                    "role": "system",
                    "content": f"""You are provided with a scientific figure and its associated caption and text. Plot {plot_num} has been identified as a Thioflavin T (ThT) fluorescence vs. time plot.

            Your task is to extract only the experimental conditions for {plot_num} that vary between legend keys or line styles (e.g., •, ○, "black triangle", "dashed red line"). Ignore conditions that are constant across all samples (e.g., shared buffer, same temperature, same ThT concentration, etc.).

            The list of variable experimental conditions to extract includes:
            - Mutation
            - Protein/mutation concentration
            - Temperature
            - pH
            - Additives (e.g., DTT, H2O2)
            - Additive concentrations (e.g., 3 mM, 64 mM)

            If an additive is mentioned with a concentration, split it into `"Additives"` and `"Additive concentrations"` accordingly. Ensure values are not merged into one field.

            If a condition is not applicable or not mentioned, set its value to an empty list.

            Return a JSON object where each key is a legend key or line style, and the value is a dictionary containing only the variable experimental conditions for that sample. Omit fields that are not variable.

            Use this JSON format:
            {{
            "•": {{
                "pH": [],
            }},
            "red_line": {{
                "Mutation": ["WT"],
                "Additives": ["NaCl"],
                "Additive concentrations": ["10 mM"]
            }},
            ...
            }}
            """,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ]

        elif message_type == "constants_extractor" and plot_num:
            user_content = [
                {
                    "type": "text",
                    "text": input_text,  # Should include caption + surrounding context
                }
            ]

            messages = [
                {
                    "role": "system",
                    "content": f"""You are provided with a scientific figure's caption and surrounding text. Plot {plot_num} has been identified as a Thioflavin T (ThT) fluorescence vs. time plot.

            Your task is to extract only the constant experimental conditions that apply across all samples in this plot.
            
            If a condition varies between samples (e.g., mutations, concentrations, additives, etc.), then its value must be set to null.

            If a condition is not applicable or not mentioned, set its value to null.

            Return a single JSON object in the following format:
            {{
            "Protein": ["string"],
            "Mutation": [null],
            "Protein/mutation concentration": [null],
            "Temperature": ["number with unit"],
            "pH": ["number"],
            "Additives": ["string"],
            "Additive concentrations": ["string with unit"]
            }}

            Ignore any experimental conditions that are not listed above. Ignore conditions that do not exactly fit the specified format (e.g. buffer is not an additive). If a condition is not applicable or not mentioned, set its value to null.

            If there is more than one additive in the list, make sure additive concentrations match.
            """,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ]

        elif message_type == "match_maker":
            user_content = [
                {
                    "type": "text",
                    "text": input_text,  # Should include dictionary and csv headers
                }
            ]

            messages = [
                {
                    "role": "system",
                    "content": f"""You are given:

            1. A dictionary where each key is a either a legend symbol or a variable descriptor (e.g., "○", "red line", "pH 3") and the value is a set of experimental conditions.

            2. A list of table column headers. Time is always the first column, followed by legend symbols or variable descriptors.

            Your task is to match each column header (excluding Time) to the correct dictionary item.

            Output a list of tuples in the format:

            [("legend_symbol", "column_header"), ...]. For example: [("•", "Black Circles"), ("pH 3", "Black Triangles")]. If matching is not possible, return False.

            Do not include any additional text or explanation.
            """,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ]

        else:
            # Error catch
            raise ValueError("Invalid message type.")

        return messages

    def use_message(
        self,
        messages: list,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        **kwargs,
    ) -> dict:
        """
        Send messages to OpenAI API and get response.

        Args:
            messages: List of message dictionaries
            model: OpenAI model to use (default: gpt-4o-mini)
            **kwargs: Additional parameters for the API call

        Returns:
            Dictionary containing the response and metadata
        """

        # Default parameters
        default_params = {
            "temperature": temperature,
            "max_tokens": 1500,  # Increased from 500 to handle complex legend mappings
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }

        # Update with any provided kwargs
        params = {**default_params, **kwargs}

        try:
            response = self.client.chat.completions.create(
                model=model, messages=messages, **params
            )

            result = {
                "success": True,
                "content": response.choices[0].message.content,
                "model": model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "finish_reason": response.choices[0].finish_reason,
                "error": None,
            }

        except Exception as e:
            result = {
                "success": False,
                "content": None,
                "model": model,
                "usage": None,
                "finish_reason": None,
                "error": str(e),
            }

        return result

    def run_model(
        self,
        text: str,
        image_path: str = None,
        analysis_type: str = "data_extractor",
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        plot_num: int = None,
        **kwargs,
    ) -> dict:
        """
        Convenience method to create message and get response in one call.

        Args:
            text: Text to analyze
            image_path: Path to image file
            analysis_type: Type of analysis ('analysis', 'extraction', 'summary', 'classification')
            model: OpenAI model to use
            tht_plot_list: List of ThT plot numbers for data extraction
            **kwargs: Additional parameters for the API call

        Returns:
            Dictionary containing the response and metadata
        """

        messages = self.create_message(text, image_path, analysis_type, plot_num)
        return self.use_message(messages, model, temperature, **kwargs)
