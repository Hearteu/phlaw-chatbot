import json

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .chat_engine import chat_with_law_bot


@csrf_exempt
def chat_api(request):
    if request.method == "POST":
        body = json.loads(request.body)
        query = body.get("query")

        if not query:
            return JsonResponse({"error": "Missing query"}, status=400)

        try:
            answer = chat_with_law_bot(query)
            return JsonResponse({"response": answer})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "POST method required"}, status=405)
