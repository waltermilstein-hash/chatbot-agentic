"""
"""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from fastapi import status
from typing import List, Optional, Tuple, Dict, Any
import logging
import json
from config.Constants import KEY_OPEN_AI

from framework.ConfigurationManager.ConfigurationManagerSelected import ConfigurationManagerSelected
from chatbot.CustomLLMRetrieverByCelery import CustomLLMRetrieverByCelery

# Logger configuration
logging.basicConfig(level=logging.INFO)
from fastapi import APIRouter, UploadFile, Form, File, HTTPException, Body
from framework.Database.MongoDb.MongoDb import MongoDBManager
api_router_processing_pages = APIRouter()
from PIL import Image, UnidentifiedImageError
from datetime import datetime

os.environ['OPENAI_API_KEY'] = KEY_OPEN_AI
logging.basicConfig()
logger = logging.getLogger(__name__)


class ChatbotTrainer:
    @staticmethod
    def training_chat_bot_buscador(
            files: Dict[str, Dict],
            json_data: dict

    ):
        try:
            if not isinstance(json_data, dict):
                raise ValueError("json_data must be a dictionary")

            # Enrich the JSON
            json_data.update({
                'document_type': 'facturas',
                'config_chatbot': 'documental'
            })

            rules = json_data
            logger.info("JSON decoded successfully.")
        except Exception as e:
            logger.error(f"Error preparing the JSON: {e}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The provided JSON is not valid.")

        # Validate required fields
        required_fields = ["user", "company", "aplicacion"]
        missing_fields = [field for field in required_fields if field not in rules]
        if missing_fields:
            msg = f"The following fields are missing in json_data: {', '.join(missing_fields)}"
            logger.error(msg)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)

        # Initialize dependencies
        try:
            initConfig = ConfigurationManagerSelected()
            config = initConfig.load_configuration(rules['aplicacion'])
            train = CustomLLMRetrieverByCelery(config)
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Internal configuration error.")

        # Register the user if it does not exist
        try:
            user_query = {
                "user": str(rules['user']).upper(),
                "company": str(rules['company']).upper()
            }
        except Exception as e:
            logger.error(f"Error validating or inserting the user: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Error handling user data.")

        # Process a single file
        try:
            file_name = files["file_name"]
            content_type ="pdf"
            contents_ = files["contents"]

            contents ={}
            contents["file_name"] = file_name
            contents["text"] = contents_

            if content_type != 'pdf':
                logger.error(f"Unsupported file type: {content_type}")
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                    detail=f"Unsupported file type: {content_type}")

            # Text extraction
            #pdf_text = train.extract_azure_with_handler_errors_sync(contents)
            if not contents:
                logger.error(f"The file {file_name} contains no processable text.")
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                    detail=f"The file {file_name} contains no processable text.")

            doc_type = train.detect_document_type_sync(contents['text'])
            # Insert the document into the LLM collection
            train.insert_document_sync(
                user=user_query["user"],
                company=user_query["company"],
                content=contents,
                doc_type=doc_type,
                file_name=file_name


            )

            # Store the original file in MongoDB
            logger.info(f"File {file_name} processed and stored successfully.")
        except Exception as e:
            logger.error(f"Error processing the file {file_name}: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"Error processing the file {file_name}.")

        return {
            "response": "collection created or updated",
            "post": rules

        }
