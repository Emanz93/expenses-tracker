import base64
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

PBKDF2_ITERATIONS = 600_000   # OWASP-recommended minimum for PBKDF2-SHA256

class WrongPasswordException(ValueError):
    """ Exception raised when the wrong password has been provided. """
    def __init__(self, *args: object):
        super().__init__(*args)


def encrypt_message(message, password, salt):
    """ This function encrypt a plain message and returns it crypted adn encoded
    Parameters:
        message: byte. Message to encrypt.
        password: String.
        salt: String.
    Returns:
        encrypted_data: String byte-encoded. Encrypted message.
    """
    if salt is not None:
        salt = bytes(salt, encoding="utf8")
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=PBKDF2_ITERATIONS
        )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    keymaster = Fernet(key)
    encrypted_data = keymaster.encrypt(message)
    return encrypted_data


def encrypt_file(filename, password, salt):
    """ This function encrypts the given file with a password provided.
    Parameters:
        filename: String. Path of the plain file to be encrypted.
        password: String. Password to be used for the encryption.
        salt: String.
    Returns:
        encrypted_filename: String. Path of the encrypted file.
    """
    # read the bytes of the plain file
    with open(filename, "rb") as f:
        data = f.read()

    encrypted_data = encrypt_message(data, password, salt)

    # write the encrypted data
    encrypted_filename = filename + ".aes"
    with open(encrypted_filename, "wb") as f:
        f.write(encrypted_data)

    return encrypted_filename


def decrypt_message(encrypted_message, password, salt):
    """ This function decrypt a plain message and returns it crypted adn encoded
    Parameters:
        encrypted_message: Byte. String byte-encoded.
        password: String.
        salt: String.
    Returns:
        decrypted_data: Byte. Decrypted message.
    Raises:
        WrongPasswordException: when the message cannot be decrypted due to wrong password.
    """
    if salt is not None:
        salt = bytes(salt, encoding="utf8")

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=PBKDF2_ITERATIONS
        )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    keymaster = Fernet(key)
    try:
        decrypted_data = keymaster.decrypt(encrypted_message)
    except InvalidToken:
        raise WrongPasswordException('Wrong password or wrong salt')
    return decrypted_data


def decrypt_file(filename, password, salt):
    """ This function decrypts the given file with a password provided.
    Parameters:
        filename: String. Path of the plain file to be decrypted.
        password: String. Password to be used for the encryption.
        salt: String. Optional.
    Returns:
        decrypted_filename: String. Path of the plain file.
    """
    with open(filename,"rb") as f:
        encrypted_data = f.read()
    
    decrypted_data = decrypt_message(encrypted_data, password, salt)

    decrypted_filename = filename[:-4] if filename.endswith('.aes') else filename + '.dec'
    with open(decrypted_filename, "wb") as f:
        f.write(decrypted_data)

    return decrypted_filename


def generate_key():
    """ Generate a key. 
    Retunrs:
        key: String.
    """
    return Fernet.generate_key().decode()

def decrypt_file_in_memory(filename, password, salt):
    """ Decrypt a file and return the plaintext bytes, without writing any file to disk.
    Parameters:
        filename: String. Path of the encrypted file.
        password: String. Password to be used for the decryption.
        salt: String.
    Returns:
        decrypted_data: Byte. Content of the decrypted file.
    """
    with open(filename, "rb") as f:
        encrypted_data = f.read()
    return decrypt_message(encrypted_data, password, salt)