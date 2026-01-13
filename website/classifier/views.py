from django.shortcuts import render, redirect
from django.contrib import messages
from .utils import predictor


def index(request):
    """Página inicial com formulário de upload"""
    return render(request, 'classifier/index.html')


def predict(request):
    """Processa a imagem e retorna a predição"""
    if request.method == 'POST':
        if 'image' not in request.FILES:
            messages.error(request, 'Nenhuma imagem foi enviada.')
            return redirect('classifier:index')

        image_file = request.FILES['image']

        if not image_file.content_type.startswith('image/'):
            messages.error(
                request, 'O arquivo enviado não é uma imagem válida.')
            return redirect('classifier:index')

        try:
            result = predictor.predict(image_file)

            return render(request, 'classifier/result.html', {
                'prediction': result
            })

        except Exception as e:
            messages.error(request, f'Erro ao processar a imagem: {str(e)}')
            return redirect('classifier:index')

    return redirect('classifier:index')
