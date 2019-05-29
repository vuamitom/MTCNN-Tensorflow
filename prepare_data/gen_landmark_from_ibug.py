import xml.etree.ElementTree as ET
import os

def gen_landmark_txt(xml_path, img_dir, output_file):
    """
    generate txt meta file according to format 
    <file_path> x1 y1 x2 y2 lm1x lm1y lm2x lm2y .... 
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    points = []
    img_shapes = []
    img_names = []
    total = 0
    invalid_count = 0
    content = []
    # lmc = 0
    for img in root.iter('image'):
        # nb = sum([1 for _ in img.iter('box')])
        # n_b = nb if nb > n_b else n_b 
        box = img.find('box')
        
        img_path = os.path.join(img_dir, img.get('file'))
        
        # xs = np.array(xs, dtype=np.int32).reshape((-1, 1))
        # ys = np.array(ys, dtype=np.int32).reshape((-1, 1))
        # points.append(np.hstack([xs, ys]))
        # points.append(np.array(xy, dtype=np.int32))
        # img_names.append(img.get('file'))
        bound = (box.get('top'), box.get('left'), box.get('width'), box.get('height'))
        bound = [int(b) for b in bound]
        invalid = any([x < 0 for x in bound])
        if invalid:
            invalid_count += 1
            continue
        line = []
        line.append(img_path)
        bbox = bound[1], bound[1] + bound[2], bound[0], bound[0] + bound[3]
        for coor in bbox:
            line.append(str(coor))

        # cc = 0
        # prev = 0
        for p in box.iter('part'):
            # n = int(p.get('name'))
            # if n < prev:
            #     print('ERROR')
            #     exit(0)
            # else:
            #     prev = n 
            line.append(p.get('x'))
            line.append(p.get('y'))
        #     cc += 1
        # if cc < 68:
        #     print ('only have ', cc , ' landmarks ')
        # elif total < 10:
        #     print (' no landmarks = ', cc)
        # lmc = cc if cc > lmc else lmc 
        content.append(' '.join(line))
        total += 1
    # print('OK')
    # exit(0)
    with open(output_file, 'w') as f:
        f.write('\n'.join(content))
        # with Image.open(path) as imgRef:
        #     w, h = imgRef.size
        #     img_shapes.append((w, h, bound))
    print('total = ', total, ' invalid = ', invalid_count)
    # return np.array(points), img_shapes, img_names



if __name__ == '__main__':
    gen_landmark_txt('/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml',
                    '/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset',
                    '/home/tamvm/Projects/MTCNN-Tensorflow/prepare_data/trainImageLandmarkList.txt')