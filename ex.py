from googletrans import Translator
translator = Translator()
print(translator.translate("Hello, how are you?", dest="es").text)
