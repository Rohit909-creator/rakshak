import base64
import os
from google import genai
from google.genai import types


def generate():
    client = genai.Client(
        # api_key=os.environ.get("GEMINI_API_KEY"),
        api_key = "",
    )
    files = [
        # Make the file available in local system working directory
        client.files.upload(file="./frames/frame27.jpg"),
        # Make the file available in local system working directory
        client.files.upload(file="./frames/frame26.jpg"),
        # Make the file available in local system working directory
        client.files.upload(file="./frames/frame25.jpg"),
        # Make the file available in local system working directory
        client.files.upload(file="./frames/frame24.jpg"),
    ]


    files2 = [
        # Make the file available in local system working directory
        client.files.upload(file="./frames/frame121.jpg"),
        # Make the file available in local system working directory
        client.files.upload(file="./frames/frame122.jpg"),
        # Make the file available in local system working directory
        client.files.upload(file="./frames/frame123.jpg"),
        # Make the file available in local system working directory
        client.files.upload(file="./frames/frame124.jpg"),
    ]
    
    model = "gemini-2.0-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(
                    file_uri=files[0].uri,
                    mime_type=files[0].mime_type,
                ),
                types.Part.from_uri(
                    file_uri=files[1].uri,
                    mime_type=files[1].mime_type,
                ),
                types.Part.from_uri(
                    file_uri=files[2].uri,
                    mime_type=files[2].mime_type,
                ),
                types.Part.from_uri(
                    file_uri=files[3].uri,
                    mime_type=files[3].mime_type,
                ),
                types.Part.from_text(
                    text="""Image:"""
                ),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(
                    text="""Image: The image shows a car rear-ending a truck on the highway. AI: Car accident. Threat level: Medium."""
                ),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text="""Image:"""
                ),
            ],
        ),
        types.Content(
            role="user",
            parts = [
                types.Part.from_uri(
                    file_uri=files2[0].uri,
                    mime_type=files2[0].mime_type,
                ),
                types.Part.from_uri(
                    file_uri=files2[1].uri,
                    mime_type=files2[1].mime_type,
                ),
                types.Part.from_uri(
                    file_uri=files2[2].uri,
                    mime_type=files2[2].mime_type,
                ),
                types.Part.from_uri(
                    file_uri=files2[3].uri,
                    mime_type=files2[3].mime_type,
                ),
                types.Part.from_text(
                    text="""Image:"""
                ),
            ]
        )
    ]
    generate_content_config = types.GenerateContentConfig(
                temperature=1,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
                response_mime_type="text/plain",
                system_instruction=[
                    types.Part.from_text(
                        text="""You are an AI powered Accident, Crime, Bullying, detection program which can alert the emergency services, or patrol services the incident that has occured.

        Parameter's possible value:
        Threat level - Very High, High, Medium, Low, None

        The structure of prompt will be like this,
        Image: <image> or [images....]
        AI: <short description> <threat level>"""
                    ),
                ],
            )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")


generate()
