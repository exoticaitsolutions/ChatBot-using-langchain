from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
import os
from .models import UploadedPDF
from .forms import UploadPDFForm
from chatbot_project import settings
from django.contrib import messages

def upload_pdf(request):
    if request.method == 'POST':
        form = UploadPDFForm(request.POST, request.FILES)
        if form.is_valid():
            pdf_name = form.cleaned_data['pdf_file'].name
            print("pdf_name : ", pdf_name)
            if UploadedPDF.objects.filter(pdf_file__iexact=pdf_name).exists():
                messages.error(request, 'PDF with the same name already exists. Please upload another PDF.')
                return redirect("/")
            else:
                form.save()
                messages.success(request, 'File uploaded successfully.')

                return redirect('/chat/')
    else:
        form = UploadPDFForm()
    return render(request, 'upload_pdf.html', {'form': form, 'messages': messages.get_messages(request)})

def chat_view(request):
    """
    Define the main view function for handling chat
    """
    # os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY


    persist_directory = "./storage"

    latest_pdf = UploadedPDF.objects.order_by('-id').first()

    if latest_pdf:
        pdf_path = latest_pdf.pdf_file.path
    else:
        return HttpResponse("Please upload pdf")

    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=10)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    
    vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_directory)
    vectordb.persist()
    data = vectordb.get("source_documents")

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(model_name='gpt-4')

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
   
    context = {}

    if request.method == 'POST':
        user_input = request.POST.get('query', '')

        query = f"###Prompt {user_input}"

        try:
            llm_response = qa(query)
            docs = llm_response.get("source_documents", [])

            for doc in docs:
                source_attribute = doc.metadata.get('source', None)
                filename = os.path.basename(source_attribute)
                print("source attribute :", source_attribute)
                print("filename :", filename)
                
            response_data = {
                'result': llm_response.get('result', ''),
                'context': llm_response.get('context', ''),
                # 'pdf_name': filename,
            }
            
            return JsonResponse(response_data)

        except Exception as err:
            context['error'] = f'Exception occurred. Please try again: {str(err)}'
            return JsonResponse(context, status=500)

    return render(request, 'chat.html', context)


