import face_recognition
import glob
import json
import numpy as np
import cv2

photo_dir = 'yearbook_data/'

class comparable_student:
    def __init__(self, fn=None, ln=None, nick=None, grade=None):
        self.fn = fn
        self.ln = ln
        self.nick = nick
        self.grade = grade

        if nick is None:
            self.nick = self.fn

    # If any entry is missing, ignore it (anything matches).
    # This has a fail case where if everything is none-ignored, everything matches.
    # Hmm, maybe I should treat nicknames as first names? Nicknames screw things up.
    def __eq__(self, other):
        fn_match = self.fn is None or other.fn is None or self.fn == other.fn
        ln_match = self.ln is None or other.ln is None or self.ln == other.ln
        grade_match = self.grade is None or other.grade is None or self.grade == other.grade

        nick_match = self.nick and other.nick and self.nick == other.nick #None and anything is None.
        name_match = fn_match or nick_match

        return name_match and ln_match and grade_match

class student_info(comparable_student):
    def __init__(self, infostr, pic_loc):
        last, _, fg = infostr.partition(', ')

        if '(' in fg and ')' in fg:
            o = fg.index('(')
            c = fg.index(')')
            self.nick = fg[o+1:c]
            fg = fg[:o-1] + fg[c+1:]
        else:
            self.nick = ''

        first, _, grade = fg.partition(' - ')
        self.ln = last
        self.fn = first
        self.grade = grade
        self.pic = pic_loc

    def __repr__(self):
        return '<student_info ln:{} fn:{} grade:{} pic:{}> at {}'.format(self.ln, self.fn, self.grade, self.pic, id(self))

    def __str__(self):
        if self.nick != '':
            return'{}, {} ({}), grade {}'.format(self.ln, self.fn, self.nick, self.grade)
        return '{}, {}, grade {}'.format(self.ln, self.fn, self.grade)

    def get_pic(self):
        return cv2.imread(self.pic)[...,::-1]


# Creates two lists- one of students info and one of the students' encodings.
# The lists are matched by index.
def gen_encodings():
    image_headers = []
    for json_header in glob.glob(photo_dir + '*.json'):
        with open(json_header) as f:
            image_headers.append(json.loads(f.read()))

    image_headers = np.concatenate(image_headers)

    namearray = []
    encarray = []
    for im in image_headers:
        rgb_im = cv2.imread(photo_dir + im['imageName'])[...,::-1]
        encarray.append(face_recognition.face_encodings(rgb_im)[0])
        namearray.append(student_info(im['name'], photo_dir + im['imageName']))
        assert len(namearray) == len(encarray)

    namearray = np.array(namearray)
    return namearray, encarray


# For if an error occurs.
# Finds all people in known_students that match the firstname, lastname, grade descriptors
def find_person(known_students, firstname=None, lastname=None, nick=None, grade=None):
    to_match = comparable_student(fn=firstname, ln=lastname, nick=nick, grade=grade)
    matches = np.argwhere(known_students == to_match)
    return matches[...,0]


# Basic usage of this code.
def main(target_im):
    target_encoding = face_recognition.face_encodings(target_im)[0]
    known_students, known_encodings = gen_encodings()

    distances = face_recognition.face_distance(known_encodings, target_encoding)
    prob_order = np.argsort(distances)

    top_five = prob_order[:5]
    print('The top five most likely candidates are:')
    for student, prob in zip(known_students[top_five], distances[top_five]):
        print('%0.3f: %s' % (prob, student))

    # show pics of the top five?
    pics = [s.get_pic() for s in known_students[top_five]]

    # pretend like the person didn't show up in top 5.
    # enter their first and last name, or grade. all optional fields.
    fn = 'Aman'
    ln = None
    gr = None
    possible_ids = find_person(known_students, firstname=fn, lastname=ln, grade=gr)
    print(known_students[possible_ids].astype(str))
    print('Your distance value was {}'.format(distances[possible_ids]))


if __name__ == '__main__':
    targ = cv2.imread('ashwin2.jpg')[:,:,::-1]
    main(targ)
