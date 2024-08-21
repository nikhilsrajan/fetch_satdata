import dataclasses


@dataclasses.dataclass
class SSHCredentials:
    hostname:str
    username:str
    password:str
