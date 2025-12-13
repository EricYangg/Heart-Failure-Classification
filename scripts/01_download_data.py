# download_data.py
# author: Mara Sanchez
# date: 2024-12-02

import click
import os
import requests

def read_file(url, directory):
    """
    # Title: Function to read a zip file from the given URL and extract its contents to the specified directory.
    # Author: Tiffany Timbers
    # Source: https://github.com/ttimbers/breast-cancer-predictor/blob/2.0.0/scripts/download_data.py
    # Taken from: breast-cancer-predictor version 2.0.0
    
    Download a CSV file from the provided URL and save it to the specified directory.

    Parameters:
    ----------
    url : str
        The URL of the csv file to be read.
    directory : str
        The directory where the contents of the csv file will be saved.

    Returns:
    -------
    None
    """
    request = requests.get(url)
    filename_from_url = os.path.basename(url)

    # check if URL exists, if not raise an error
    if request.status_code != 200:
        raise ValueError('The URL provided does not exist.')
    
    # check if the URL points to a csv file, if not raise an error  
    #if request.headers['content-type'] != 'application/csv':
    if filename_from_url[-4:] != '.csv':
        raise ValueError('The URL provided does not point to a csv file.')
    
    # check if the directory exists, if not raise an error
    if not os.path.isdir(directory):
        raise ValueError('The directory provided does not exist.')

    # write the csv file to the directory
    path_to_csv_file = os.path.join(directory, filename_from_url)
    with open(path_to_csv_file, 'wb') as f:
        f.write(request.content)

@click.command()
@click.option('--url', type=str, help="URL of dataset to be downloaded")
@click.option('--write_to', type=str, help="Path to directory where raw data will be written to")
def main(url, write_to):
    """Downloads CSV data from the web to a local filepath."""
    try:
        read_file(url, write_to)
    except:
        os.makedirs(write_to)
        read_file(url, write_to)

if __name__ == '__main__':
    main()