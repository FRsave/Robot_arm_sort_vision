import json

filename = "objects.json"

class Object:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, number, centreX, centreY,point1, point2, point3, point4, ang, w, h, object_type ):
        self.number = number
        self.centreX = centreX
        self.centreY = centreY
        self.point1 = point1
        self.point2 = point2
        self.point3 = point3
        self.point4 = point4
        self.angle = ang
        self.width = w
        self.height = h
        self.type = object_type





#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def print_info(self):
     print(self.number, self.centreX, self.centreY, self.point1, self.point2, self.point3, self.point4, self.angle, self.width,self.height, self.type)

    def save_to_json(self, filename):
        object_dict = ({"objects" :[ {'object num': self.number, 'centre X': self.centreX,'centre Y': self.centreY,'p1': self.point1,
                       'p2':self.point2,'p3': self.point3,'p4':self.point4, 'angle': self.angle, 'width': self.width, 'height':self.height, 'object type': self.type}]})
        with open(filename, 'w') as f:
            f.write(json.dumps(object_dict, indent= 1,separators=(',',': ')))


    def load_from_file(self, filename):
        with open(filename,'r') as f:
            data = json.load(f.read())

        self.number = data['object num']
        self.centreX = data['centreX']
        self.centreY = data['centreY']
        self.point1 = data['point1']
        self.point2 = data['point2']
        self.point3 = data['point3']
        self.point4 = data['point4']
        self.angle = data['angle']
        self.width = data['width']
        self.height = data['height']
        self.type = data['object type']


    def add_to_file(self, filename, number, centreX, centreY,point1, point2, point3, point4, angle,w,h, object_type):

        with open(filename) as f:
            dic = json.load(f)

        x = {'object num': number, "centre X": centreX, "centre Y": centreY, 'p1': point1, 'p2': point2, 'p3': point3,
             'p4': point4, 'angle': angle, 'width': w, 'height':h, 'object type': object_type}

        dic["objects"].append(x)

        with open(filename, 'w') as f:
            json.dump(dic, f, indent=1, separators=(',', ': '))










