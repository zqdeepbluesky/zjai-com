class im_info(object):
    def __init__(self, name='', path='', image_extension='.jpg', image_bgr=None):
        self.name = name # not include extension
        self.path = path
        self.image_extension = image_extension
        self.image_bgr = image_bgr
        self.width=0
        self.height=0
        self.channel=3