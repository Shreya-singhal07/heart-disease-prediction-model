# security.py
"""
Security module for HIPAA-compliant data protection.
Implements encryption and security protocols for healthcare data.
"""

import os
import json
import hashlib
import logging
import datetime
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64

class HealthDataSecurity:
    def __init__(self, config_path='security_config.json'):
        """Initialize the security manager with configuration."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        
    def _load_config(self, config_path):
        """Load security configuration from JSON file."""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Default security configuration
            default_config = {
                'key_directory': 'keys',
                'log_directory': 'logs',
                'encryption_iterations': 100000,
                'user_access_levels': {
                    'admin': ['read', 'write', 'delete', 'encrypt', 'decrypt'],
                    'doctor': ['read', 'write'],
                    'researcher': ['read'],
                    'analyst': ['read']
                }
            }
            # Create the default config file
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            return default_config
    
    def setup_logging(self):
        """Configure logging for security events."""
        log_dir = self.config.get('log_directory', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f'security_{datetime.datetime.now().strftime("%Y%m%d")}.log')
        
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('health_security')
    
    def generate_key(self, password, salt=None):
        """
        Generate an encryption key from a password.
        
        Args:
            password: Password string
            salt: Optional salt bytes, generated if not provided
            
        Returns:
            Tuple containing (key, salt)
        """
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.config.get('encryption_iterations', 100000),
            backend=default_backend()
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        
        return key, salt
    
    def save_key(self, key, filename, user_id=None):
        """
        Save an encryption key to a file.
        
        Args:
            key: The encryption key
            filename: Destination filename
            user_id: Optional user identifier for auditing
        """
        key_dir = self.config.get('key_directory', 'keys')
        os.makedirs(key_dir, exist_ok=True)
        
        key_path = os.path.join(key_dir, filename)
        with open(key_path, 'wb') as f:
            f.write(key)
        
        self.logger.info(f"Key saved to {key_path}" + (f" by user {user_id}" if user_id else ""))
    
    def load_key(self, filename, user_id=None):
        """
        Load an encryption key from a file.
        
        Args:
            filename: Source filename
            user_id: Optional user identifier for auditing
            
        Returns:
            The encryption key
        """
        key_dir = self.config.get('key_directory', 'keys')
        key_path = os.path.join(key_dir, filename)
        
        if not os.path.exists(key_path):
            self.logger.error(f"Key file not found: {key_path}")
            raise FileNotFoundError(f"Key file not found: {key_path}")
        
        with open(key_path, 'rb') as f:
            key = f.read()
        
        self.logger.info(f"Key loaded from {key_path}" + (f" by user {user_id}" if user_id else ""))
        return key
    
    def encrypt_file(self, input_file, output_file, key, user_id=None):
        """
        Encrypt a file using the provided key.
        
        Args:
            input_file: Path to the file to encrypt
            output_file: Path where the encrypted file will be saved
            key: Encryption key
            user_id: Optional user identifier for auditing
            
        Returns:
            Path to the encrypted file
        """
        fernet = Fernet(key)
        
        with open(input_file, 'rb') as f:
            data = f.read()
        
        encrypted_data = fernet.encrypt(data)
        
        with open(output_file, 'wb') as f:
            f.write(encrypted_data)
        
        self.logger.info(f"File encrypted: {input_file} -> {output_file}" + 
                         (f" by user {user_id}" if user_id else ""))
        
        return output_file
    
    def decrypt_file(self, input_file, output_file, key, user_id=None):
        """
        Decrypt a file using the provided key.
        
        Args:
            input_file: Path to the encrypted file
            output_file: Path where the decrypted file will be saved
            key: Encryption key
            user_id: Optional user identifier for auditing
            
        Returns:
            Path to the decrypted file
        """
        fernet = Fernet(key)
        
        with open(input_file, 'rb') as f:
            encrypted_data = f.read()
        
        try:
            decrypted_data = fernet.decrypt(encrypted_data)
            
            with open(output_file, 'wb') as f:
                f.write(decrypted_data)
            
            self.logger.info(f"File decrypted: {input_file} -> {output_file}" + 
                             (f" by user {user_id}" if user_id else ""))
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"Decryption failed: {str(e)}")
            raise
    
    def check_access(self, user_id, user_role, operation):
        """
        Check if a user has permission for an operation.
        
        Args:
            user_id: User identifier
            user_role: User role (admin, doctor, researcher, etc.)
            operation: Requested operation (read, write, delete, etc.)
            
        Returns:
            Boolean indicating if access is allowed
        """
        user_access_levels = self.config.get('user_access_levels', {})
        allowed_operations = user_access_levels.get(user_role, [])
        
        access_granted = operation in allowed_operations
        
        log_message = (f"Access {'granted' if access_granted else 'denied'} - "
                       f"User: {user_id}, Role: {user_role}, Operation: {operation}")
        
        if access_granted:
            self.logger.info(log_message)
        else:
            self.logger.warning(log_message)
        
        return access_granted
    
    def hash_pii(self, value):
        """
        Create a secure hash of personally identifiable information.
        
        Args:
            value: The PII value to hash
            
        Returns:
            Secure hash of the input value
        """
        # Use SHA-256 for secure hashing
        return hashlib.sha256(str(value).encode()).hexdigest()