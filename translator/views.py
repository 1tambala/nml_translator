from django.shortcuts import render
from translator_data.eng_translator import evaluate, normalizeString

def index(request):
    if request.method == "POST":

        word = normalizeString(request.POST.get("word"))
       
        try:
            results = evaluate(word)
            result = ""
            for re in results:
                result += ' ' + re
        except KeyError:
            result = None

        if result is not None:
            data = {
                'word': word,
                'translation': str(result),
            }
            return render(request, 'index.html', data)
        else:
            result = "Pepani! Palibe matanthauzo omwe apezeka ku mawu omwe munalemba"
            data = {
                'word': word,
                'translation': result,
            }
            return render(request, 'index.html', data)
    else:
        return render(request, 'index.html')


def about(request):
    return render(request, 'about.html')
