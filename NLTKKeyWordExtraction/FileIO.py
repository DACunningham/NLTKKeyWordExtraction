class FileIO(object):
    """Exposes methods to load single or multiple text files into memory"""

    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)


    def load_text_file(self, file_location):
        """Loads text file and returns string contents to caller. 'file_location' is file location and file name to be loaded """
        file = open(file_location, 'rt', encoding = "utf8")
        fileText = file.read()
        file.close()
        return fileText

    def load_many_text_files(self, new_file_paths):
        """Load many text files as string into an array.  new_file_paths is an array of file locations and file names"""
        documentLibrary = []
        for filePath in new_file_paths:
            documentLibrary.append(self.load_text_file(filePath))
        return documentLibrary

