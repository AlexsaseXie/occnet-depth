from fileinput import filename
from ntpath import join
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators import csrf
from . import settings
import os
import numpy as np
import time
 
def hello(request):
    return HttpResponse("Hello world ! ")

def runnoob(request):
    context          = {}
    context['hello'] = 'Hello World!'
    return render(request, 'runnoob.html', context)

def search_form(request):
    return render(request, 'search_form.html')
 
def search(request):  
    request.encoding='utf-8'
    if 'q' in request.GET and request.GET['q']:
        message = '你搜索的内容为: ' + request.GET['q']
    else:
        message = '你提交了空表单'
    return HttpResponse(message)

def search_post(request):
    ctx ={}
    if request.POST:
        ctx['rlt'] = request.POST['q']
    return render(request, "post.html", ctx)

# work area
def section3_view(request):
    if request.method == 'GET':
        return render(request, 'section3.html')

def section4_view(request):
    if request.method == 'GET':
        return render(request, 'section4.html')

def receive_view(request):
    if request.method == 'POST':
        # 取文件数据
        a_file = request.FILES['file']
        print("上传文件名是：", a_file.name)
        # 拼接存储绝对路径
        lst = a_file.name.split('.')
        lst[-2] += '_%d' % time.time()
        filename = '.'.join(lst)

        save_path = os.path.join(settings.MEDIA_ROOT, filename)
        with open(save_path, 'wb') as f:
            # a_file.file 文件数据
            # a_file.file.read() 读出来
            data = a_file.file.read()
            f.write(data)
        return HttpResponse("media/" + filename)

def section3_example(request):
    TEST = False
    if request.method == 'GET':
        if TEST:
            root = os.path.join(settings.MEDIA_ROOT, 'section3')
            filenames = os.listdir(root)

            url_root = '/media/section3/'
            data = []
            for info in filenames:
                if os.path.isfile(os.path.join(root, info)):
                    data.append({
                        'img': url_root + '/' + info, 
                        'mask':  url_root + '/' + info,
                        'Rt': [0.1],
                        'K': [0.1],
                    })
        else:
            root = os.path.join(settings.MEDIA_ROOT, 'section3_gathered')
            classes = os.listdir(root)

            data = []
            for c in classes:
                class_root = os.path.join(root, c)
                modelnames = sorted(os.listdir(class_root))[:12]

                url_root = '/media/section3_gathered'
                for modelname in modelnames:
                    model_root = os.path.join(class_root, modelname)
                    camera_file = os.path.join(model_root, 'cameras.npz')

                    camera_dict = np.load(camera_file)
                    Rt = camera_dict['world_mat_%d' % 0][:3,:4].flatten().tolist()
                    K = camera_dict['camera_mat_%d' % 0][:3,:3]
                    K[:2,:] *= 224.0/137.0
                    K = [K[0,0], K[0,2], K[1,1], K[1,2]]
                    data.append({
                        'img': url_root + '/' + c + '/' + modelname + '/rgb.png',
                        'mask': url_root + '/' + c + '/' + modelname + '/mask.png',
                        'Rt': Rt,
                        'K': K,
                        'output': url_root + '/' + c + '/' + modelname + '/ours.obj',
                        'gt': url_root + '/' + c + '/' + modelname + '/gt.obj'
                    })
            

        return JsonResponse(data, safe=False)

def section3_predict(request):
    if request.method == 'POST':
        url_root = '/media/section3'
        data = {
            'output': url_root + '/case01/output.obj'
        }

        return JsonResponse(data)

def section4_example(request):
    if request.method == 'GET':
        TEST = False
        if TEST:
            root = os.path.join(settings.MEDIA_ROOT, 'section4')
            filenames = os.listdir(root)

            url_root = '/media/section4'
            data = [ 
                {
                    'img': url_root + '/' + info + '/vis.png', 
                    'gt':  url_root + '/' + info + '/gt.obj',
                    'pc': url_root + '/' + info + '/pc.obj',
                    'sail_s3': url_root + '/' + info + '/output.obj',
                } for info in filenames
            ]
        else:
            root = os.path.join(settings.MEDIA_ROOT, 'section4_gathered')
            classes = os.listdir(root)

            data = []
            for c in classes:
                class_root = os.path.join(root, c)
                modelnames = sorted(os.listdir(class_root))

                url_root = '/media/section4_gathered/'
                for modelname in modelnames:
                    data.append({
                        'img': url_root + '/' + c + '/' + modelname + '/gt_img.png',
                        'gt':  url_root + '/' + c + '/' + modelname + '/gt.obj',
                        'pc': url_root + '/' + c + '/' + modelname + '/input_pc_30000.obj',
                        'pc_download': url_root + '/' + c + '/' + modelname + '/input_pc_30000.npz',
                        'sal': url_root + '/' + c + '/' + modelname + '/sal.obj',
                        'sail_s3': url_root + '/' + c + '/' + modelname + '/sail_s3.obj',
                    })

        return JsonResponse(data, safe=False)