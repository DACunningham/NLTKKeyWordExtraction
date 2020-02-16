class FileIO(object):
    """description of class"""

    #filePaths = []
    documentLibrary = []
    processedDocLibrary = []

    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)


    def loadTextFile(self, fileLocation):
        file = open(fileLocation, 'rt', encoding = "utf8")
        fileText = file.read()
        file.close()
        return fileText

    def load_all_files(self, newFilePaths):
        for filePath in newFilePaths:
            self.documentLibrary.append(self.loadTextFile(filePath))
        return

