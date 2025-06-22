from django.shortcuts import render
from django.http import JsonResponse
from recommender import ml_model

# Create your views here.
def run_page(request):        
    if request.method == 'POST':
        message = request.POST.get('message')
        message = message.replace('\n','').replace(' ','').split(',')
        message.remove('Herearethelistofcourses')
        message = list(filter(lambda x: x!='', message))

        '''with open('SS', 'w') as file:
            file.write(str(message))'''
        
        response = ml_model.get_courses(*message[:3], message[3:])

        return JsonResponse({'message': "MESSAGE", 'response': response})
    return render(request, 'FORM.html')
