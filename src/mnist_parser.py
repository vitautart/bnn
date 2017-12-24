
class mnistParser:

    def __init__ (self, labelfile, imagefile):
        self.lfname = labelfile
        self.imfname = imagefile
        self.imf = None
        self.imfmagic = None
        self.imfnumber = None
        self.imfrows = None
        self.imfcoll = None
        self.lf = None
        self.lfmagic = None
        self.lfnumber = None

    def open (self):
        try:
            self.lf = open(self.lfname, 'rb')
            self.imf = open(self.imfname, 'rb')
        except IOError:
            print("Couldn't find file")
            raise

    def close (self):
        self.lf.close()
        self.imf.close()

    def parse_labels (self):
        labels = []
        i = 0
        value = None
        vectorized_label = []
        self.lfmagic = int.from_bytes(self.lf.read(4), byteorder = 'big')
        if (self.lfmagic == 2049):
            self.lf.seek(4)
            self.lfnumber = int.from_bytes(self.lf.read(4), byteorder = 'big')
            while i<self.lfnumber:
                #print(self.lf.tell())
                self.lf.seek(0, 1) # move pointer to next position in file
                value = int.from_bytes(self.lf.read(1), byteorder = 'big')
                vectorized_label = []
                for k in range(10):
                    if (k == value):
                        vectorized_label.append(1.0)
                    else:
                        vectorized_label.append(0.0)
                labels.append (vectorized_label)
                i=i+1
        return labels

    def parse_img_2 (self): #Parse MNIST images to list of 2D-arrays. Just auxiliary function
        imgs = []
        pic = []
        row_pic = []
        row_bytes = None
        i = 0
        row_i = 0
        self.imfmagic = int.from_bytes(self.imf.read(4), byteorder = 'big')
        if (self.imfmagic == 2051):
            self.imf.seek(4)
            self.imfnumber = int.from_bytes(self.imf.read(4), byteorder = 'big')
            self.imf.seek(8)
            self.imfrows = int.from_bytes(self.imf.read(4), byteorder = 'big')
            self.imf.seek(12)
            self.imfcoll = int.from_bytes(self.imf.read(4), byteorder = 'big')
            #self.imf.seek(4, 1)
            while i < self.imfnumber:
                pic = []
                for row_i in range(self.imfrows):
                    self.imf.seek(0, 1)
                    row_bytes = self.imf.read(28)
                    row_pic = []
                    for pixel in row_bytes:
                        row_pic.append(pixel)
                    pic.append(row_pic)
                imgs.append(pic)
                i = i+1
        return imgs

    def parse_img (self):
        imgs = []
        pic = []
        row_bytes = None
        i = 0
        row_i = 0
        self.imfmagic = int.from_bytes(self.imf.read(4), byteorder = 'big')
        if (self.imfmagic == 2051):
            self.imf.seek(4)
            self.imfnumber = int.from_bytes(self.imf.read(4), byteorder = 'big')
            self.imf.seek(8)
            self.imfrows = int.from_bytes(self.imf.read(4), byteorder = 'big')
            self.imf.seek(12)
            self.imfcoll = int.from_bytes(self.imf.read(4), byteorder = 'big')
            while i < self.imfnumber:
                pic = []
                self.imf.seek(0, 1)
                row_bytes = self.imf.read(self.imfrows*self.imfcoll)
                for pixel in row_bytes:
                    pic.append(pixel/255.0)
                imgs.append(pic)
                i = i+1
        return imgs
