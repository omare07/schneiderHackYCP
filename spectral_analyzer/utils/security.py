"""
Security utilities for API key encryption and secure storage.

Provides secure encryption/decryption of API keys using machine-specific
salt and user-provided passwords for enhanced security.
"""

import logging
import platform
import hashlib
import base64
from typing import Optional, Dict, Any
from pathlib import Path
import json

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import keyring


class SecureKeyManager:
    """
    Secure API key management system.
    
    Features:
    - Machine-specific encryption
    - Keyring integration for secure storage
    - Password-based key derivation
    - Multiple storage backends
    """
    
    def __init__(self):
        """Initialize the secure key manager."""
        self.logger = logging.getLogger(__name__)
        
        # Application identifier for keyring
        self.app_name = "SpectralAnalyzer"
        
        # Machine-specific salt
        self._machine_salt = self._generate_machine_salt()
        
        # Storage backends (in order of preference)
        self.storage_backends = [
            self._keyring_storage,
            self._file_storage
        ]
        
        self.logger.debug("Secure key manager initialized")
    
    def _generate_machine_salt(self) -> bytes:
        """Generate machine-specific salt for key derivation."""
        try:
            # Combine machine-specific information
            machine_info = {
                'node': platform.node(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'system': platform.system()
            }
            
            # Create deterministic salt from machine info
            machine_str = json.dumps(machine_info, sort_keys=True)
            salt = hashlib.sha256(machine_str.encode()).digest()[:16]
            
            self.logger.debug("Generated machine-specific salt")
            return salt
            
        except Exception as e:
            self.logger.warning(f"Failed to generate machine salt: {e}")
            # Fallback to static salt (less secure)
            return b'spectral_analyzer_'
    
    def _derive_key(self, password: str) -> bytes:
        """
        Derive encryption key from password and machine salt.
        
        Args:
            password: User password or machine identifier
            
        Returns:
            Derived encryption key
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self._machine_salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def encrypt_api_key(self, api_key: str, password: Optional[str] = None) -> str:
        """
        Encrypt API key with password and machine binding.
        
        Args:
            api_key: API key to encrypt
            password: Optional password (uses machine ID if not provided)
            
        Returns:
            Encrypted API key as base64 string
        """
        try:
            if password is None:
                # Use machine-specific password
                password = self._get_machine_password()
            
            # Derive encryption key
            derived_key = self._derive_key(password)
            
            # Encrypt API key
            cipher = Fernet(derived_key)
            encrypted_key = cipher.encrypt(api_key.encode())
            
            # Return as base64 string
            return base64.urlsafe_b64encode(encrypted_key).decode()
            
        except Exception as e:
            self.logger.error(f"Failed to encrypt API key: {e}")
            raise
    
    def decrypt_api_key(self, encrypted_key: str, password: Optional[str] = None) -> str:
        """
        Decrypt API key with password and machine binding.
        
        Args:
            encrypted_key: Encrypted API key as base64 string
            password: Optional password (uses machine ID if not provided)
            
        Returns:
            Decrypted API key
        """
        try:
            if password is None:
                # Use machine-specific password
                password = self._get_machine_password()
            
            # Derive encryption key
            derived_key = self._derive_key(password)
            
            # Decrypt API key
            cipher = Fernet(derived_key)
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_key.encode())
            decrypted_key = cipher.decrypt(encrypted_bytes)
            
            return decrypted_key.decode()
            
        except Exception as e:
            self.logger.error(f"Failed to decrypt API key: {e}")
            raise
    
    def _get_machine_password(self) -> str:
        """Get machine-specific password for encryption."""
        try:
            # Combine multiple machine identifiers
            identifiers = [
                platform.node(),
                platform.machine(),
                str(self._machine_salt.hex())
            ]
            
            machine_password = hashlib.sha256('|'.join(identifiers).encode()).hexdigest()
            return machine_password
            
        except Exception as e:
            self.logger.warning(f"Failed to generate machine password: {e}")
            return "default_machine_password"
    
    def set_api_key(self, provider: str, api_key: str, password: Optional[str] = None) -> bool:
        """
        Store encrypted API key securely.
        
        Args:
            provider: API provider name
            api_key: API key to store
            password: Optional encryption password
            
        Returns:
            True if stored successfully
        """
        try:
            # Encrypt the API key
            encrypted_key = self.encrypt_api_key(api_key, password)
            
            # Try storage backends in order
            for storage_backend in self.storage_backends:
                try:
                    if storage_backend(provider, encrypted_key, 'set'):
                        self.logger.info(f"API key stored for provider: {provider}")
                        return True
                except Exception as e:
                    self.logger.warning(f"Storage backend failed: {e}")
                    continue
            
            self.logger.error("All storage backends failed")
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to set API key: {e}")
            return False
    
    def get_api_key(self, provider: str, password: Optional[str] = None) -> Optional[str]:
        """
        Retrieve and decrypt API key.
        
        Args:
            provider: API provider name
            password: Optional decryption password
            
        Returns:
            Decrypted API key or None if not found
        """
        try:
            # Try storage backends in order
            for storage_backend in self.storage_backends:
                try:
                    encrypted_key = storage_backend(provider, None, 'get')
                    if encrypted_key:
                        # Decrypt and return
                        api_key = self.decrypt_api_key(encrypted_key, password)
                        self.logger.debug(f"Retrieved API key for provider: {provider}")
                        return api_key
                except Exception as e:
                    self.logger.warning(f"Storage backend failed: {e}")
                    continue
            
            self.logger.info(f"No API key found for provider: {provider}")
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get API key: {e}")
            return None
    
    def delete_api_key(self, provider: str) -> bool:
        """
        Delete stored API key.
        
        Args:
            provider: API provider name
            
        Returns:
            True if deleted successfully
        """
        try:
            success = False
            
            # Try all storage backends
            for storage_backend in self.storage_backends:
                try:
                    if storage_backend(provider, None, 'delete'):
                        success = True
                except Exception as e:
                    self.logger.warning(f"Storage backend delete failed: {e}")
            
            if success:
                self.logger.info(f"API key deleted for provider: {provider}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to delete API key: {e}")
            return False
    
    def _keyring_storage(self, provider: str, encrypted_key: Optional[str], operation: str) -> Optional[str]:
        """
        Keyring-based storage backend.
        
        Args:
            provider: API provider name
            encrypted_key: Encrypted key (for set operation)
            operation: 'set', 'get', or 'delete'
            
        Returns:
            Encrypted key for get operation, success status for others
        """
        try:
            service_name = f"{self.app_name}-{provider}"
            username = "api_key"
            
            if operation == 'set':
                keyring.set_password(service_name, username, encrypted_key)
                return True
                
            elif operation == 'get':
                return keyring.get_password(service_name, username)
                
            elif operation == 'delete':
                try:
                    keyring.delete_password(service_name, username)
                    return True
                except keyring.errors.PasswordDeleteError:
                    return False
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Keyring storage failed: {e}")
            raise
    
    def _file_storage(self, provider: str, encrypted_key: Optional[str], operation: str) -> Optional[str]:
        """
        File-based storage backend (fallback).
        
        Args:
            provider: API provider name
            encrypted_key: Encrypted key (for set operation)
            operation: 'set', 'get', or 'delete'
            
        Returns:
            Encrypted key for get operation, success status for others
        """
        try:
            # Storage directory
            storage_dir = Path.home() / ".spectral_analyzer" / "keys"
            storage_dir.mkdir(parents=True, exist_ok=True)
            
            # Key file path
            key_file = storage_dir / f"{provider}.key"
            
            if operation == 'set':
                # Store encrypted key in file
                with open(key_file, 'w') as f:
                    json.dump({
                        'encrypted_key': encrypted_key,
                        'provider': provider,
                        'created': str(Path.ctime(Path.now()))
                    }, f)
                
                # Set restrictive permissions (Unix-like systems)
                try:
                    key_file.chmod(0o600)
                except Exception:
                    pass  # Windows doesn't support chmod
                
                return True
                
            elif operation == 'get':
                if key_file.exists():
                    with open(key_file, 'r') as f:
                        data = json.load(f)
                    return data.get('encrypted_key')
                return None
                
            elif operation == 'delete':
                if key_file.exists():
                    key_file.unlink()
                    return True
                return False
            
            return None
            
        except Exception as e:
            self.logger.debug(f"File storage failed: {e}")
            raise
    
    def list_stored_providers(self) -> list:
        """
        List all providers with stored API keys.
        
        Returns:
            List of provider names
        """
        providers = set()
        
        try:
            # Check keyring storage
            try:
                # This is implementation-specific and may not work on all systems
                pass
            except Exception:
                pass
            
            # Check file storage
            try:
                storage_dir = Path.home() / ".spectral_analyzer" / "keys"
                if storage_dir.exists():
                    for key_file in storage_dir.glob("*.key"):
                        provider = key_file.stem
                        providers.add(provider)
            except Exception as e:
                self.logger.warning(f"Failed to list file storage: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to list providers: {e}")
        
        return list(providers)
    
    def validate_api_key_format(self, api_key: str, provider: str) -> bool:
        """
        Validate API key format for specific provider.
        
        Args:
            api_key: API key to validate
            provider: API provider name
            
        Returns:
            True if format appears valid
        """
        try:
            if not api_key or not isinstance(api_key, str):
                return False
            
            # Basic length and character checks
            if len(api_key) < 10:
                return False
            
            # Provider-specific validation
            if provider.lower() == 'openrouter':
                # OpenRouter keys typically start with 'sk-or-'
                return api_key.startswith('sk-or-') and len(api_key) > 20
                
            elif provider.lower() == 'openai':
                # OpenAI keys typically start with 'sk-'
                return api_key.startswith('sk-') and len(api_key) > 20
                
            elif provider.lower() == 'anthropic':
                # Anthropic keys have specific format
                return len(api_key) > 30
            
            # Generic validation for unknown providers
            return len(api_key) >= 20 and api_key.isalnum()
            
        except Exception as e:
            self.logger.warning(f"API key validation failed: {e}")
            return False
    
    def get_security_info(self) -> Dict[str, Any]:
        """
        Get security configuration information.
        
        Returns:
            Dictionary with security information
        """
        return {
            'encryption_enabled': True,
            'machine_binding': True,
            'storage_backends': len(self.storage_backends),
            'keyring_available': self._is_keyring_available(),
            'file_storage_path': str(Path.home() / ".spectral_analyzer" / "keys")
        }
    
    def _is_keyring_available(self) -> bool:
        """Check if keyring storage is available."""
        try:
            # Test keyring functionality
            test_service = f"{self.app_name}-test"
            keyring.set_password(test_service, "test", "test")
            result = keyring.get_password(test_service, "test")
            keyring.delete_password(test_service, "test")
            return result == "test"
        except Exception:
            return False