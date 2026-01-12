import os
import yaml
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class ConfigLoader:
    def __init__(self, config_path: str = "config/providers.yaml"):
        self.config_path = config_path

    def load_providers(self) -> List[Dict[str, Any]]:
        """
        Load providers from YAML file or environment variables.
        """
        providers = []

        # 1. Try loading from YAML
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    data = yaml.safe_load(f)
                    if data and "providers" in data:
                        providers.extend(data["providers"])
                        logger.info(
                            f"Loaded {len(providers)} providers from {self.config_path}"
                        )
            except Exception as e:
                logger.error(f"Failed to load config from {self.config_path}: {e}")

        # 2. Add G4F default if list is empty
        if not providers:
            logger.info("No providers configured. Using default G4F provider.")
            providers.append(
                {
                    "name": "g4f-default",
                    "type": "g4f",
                    "models": ["gpt-4", "gpt-3.5-turbo", "*"],
                }
            )

        return providers
