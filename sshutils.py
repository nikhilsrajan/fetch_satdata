import os
import paramiko
import functools
import multiprocessing as mp
import tqdm

import sshcreds


def download_file_from_cluster(
    sshcreds:sshcreds.SSHCredentials,
    remotepath:str,
    download_filepath:str = None,
    download_folderpath:str = None,
    enable_auto_add_policy:bool = True, # Trust all policy, perhaps best to keep it optional
    overwrite:bool = False,
):
    if download_folderpath is None and download_filepath is None:
        raise Exception(
            "Either 'download_folderpath' or 'download_filepath' " + \
            "should be non None."
        )

    if download_filepath is None:
        filename = remotepath.split('/')[-1]
        download_filepath = os.path.join(
            download_folderpath, filename,
        )
    else:
        download_folderpath = os.path.split(download_filepath)[0]
    
    os.makedirs(download_folderpath, exist_ok=True)

    if not os.path.exists(download_filepath) or overwrite:
        temp_download_filepath = download_filepath + '.temp'

        # https://medium.com/@keagileageek/paramiko-how-to-ssh-and-file-transfers-with-python-75766179de73
        ssh_client = paramiko.SSHClient()

        if enable_auto_add_policy:
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        ssh_client.connect(
            hostname = sshcreds.hostname,
            username = sshcreds.username,
            password = sshcreds.password,
        )

        ftp_client = ssh_client.open_sftp()

        ftp_client.get(
            remotepath = remotepath,
            localpath = temp_download_filepath,
        )

        ftp_client.close()

        if os.path.exists(download_filepath):
            os.remove(download_filepath)

        os.rename(temp_download_filepath, download_filepath)
    
    return download_filepath


def _download_file_from_cluster_by_tuple(
    remotepath_download_filepath_tuple:tuple[str, str],
    sshcreds:sshcreds.SSHCredentials,
    overwrite:bool = False,
):
    remotepath, download_filepath = remotepath_download_filepath_tuple

    download_file_from_cluster(
        sshcreds = sshcreds,
        remotepath = remotepath,
        download_filepath = download_filepath,
        enable_auto_add_policy = True,
        overwrite = overwrite,
    )
    download_success = os.path.exists(download_filepath)

    return download_success


def download_files_from_cluster(
    sshcreds:sshcreds.SSHCredentials,
    remotepaths:list[str],
    download_filepaths:list[str],
    overwrite:bool = False,
    njobs:int = 16,
):
    if len(remotepaths) != len(download_filepaths):
        raise ValueError('Size of remotepaths and download_filepaths do not match.')
    
    download_file_from_cluster_by_tuple_partial = functools.partial(
        _download_file_from_cluster_by_tuple,
        sshcreds = sshcreds,
        overwrite = overwrite,
    )

    remotepath_download_filepath_tuples = list(zip(remotepaths, download_filepaths))

    with mp.Pool(njobs) as p:
        download_successes = list(tqdm.tqdm(
            p.imap(download_file_from_cluster_by_tuple_partial, remotepath_download_filepath_tuples), 
            total=len(remotepath_download_filepath_tuples)
        ))
    
    return download_successes

