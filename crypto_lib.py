import base64
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class WrongPasswordException(ValueError):
    """ Exception raised when the wrong password has been provided. """
    def __init__(self, *args: object):
        super().__init__(*args)


def encrypt_message(message, password):
    """ This function encrypt a plain message and returns it crypted adn encoded
    Parameters:
        message: String. Message to encrypt.
        password: String.
    Returns:
        encrypted_data: String byte-encoded. Encrypted message.
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b'saltybiscuits',
        iterations=480000
        )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    keymaster = Fernet(key)
    encrypted_data = keymaster.encrypt(message.encode())
    return encrypted_data


def encrypt_file_pw(filename, password):
    """ This function encrypts the given file with a password provided.
    Parameters:
        filename: String. Path of the plain file to be encrypted.
        password: String. Password to be used for the ecryption.
    Returns:
        encrypted_filename: String. Path of the encrypted file.
    """
    # read the bytes of the plain file
    with open(filename, "r") as f:
        data = f.read()

    encrypted_data = encrypt_message(data, password)

    # write the encrypted data
    encrypted_filename = filename + ".aes"
    with open(encrypted_filename, "wb") as f:
        f.write(encrypted_data)

    return encrypted_filename


def decrypt_message(encripted_message, password):
    """ This function decrypt a plain message and returns it crypted adn encoded
    Parameters:
        encripted_message: String byte-encoded.
        password: String.
    Returns:
        decripted_data: String. Encrypted message.
    Raises:
        WrongPasswordException: when the message cannot be decrypted due to wrong password.
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b'saltybiscuits',
        iterations=480000
        )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    keymaster = Fernet(key)
    try:
        decripted_data = keymaster.decrypt(encripted_message)
    except InvalidToken:
        raise WrongPasswordException('Wrong password')
    return decripted_data.decode()


def decrypt_file_pw(filename, password):
    """ This function decrypts the given file with a password provided.
    Parameters:
        filename: String. Path of the plain file to be decrypted.
        password: String. Password to be used for the ecryption.
    Returns:
        decrypted_filename: String. Path of the plain file.
    """
    with open(filename,"rb") as f:
        encrypted_data = f.read()
    
    decrypted_data = decrypt_message(encrypted_data, password)
    
    decrypted_filename = filename.replace('.aes', '')
    with open(decrypted_filename, "w") as f:
        f.write(decrypted_data)

    return decrypted_filename
