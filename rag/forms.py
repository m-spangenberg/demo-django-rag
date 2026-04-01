from django import forms


class UploadDocumentForm(forms.Form):
    title = forms.CharField(
        max_length=255,
        widget=forms.TextInput(attrs={"placeholder": "Document title"}),
    )
    pdf = forms.FileField(help_text="Upload a PDF file for ingestion.")

    def clean_pdf(self):
        pdf = self.cleaned_data["pdf"]
        if not pdf.name.lower().endswith(".pdf"):
            raise forms.ValidationError("Only PDF files are supported.")
        return pdf


class ChatQueryForm(forms.Form):
    question = forms.CharField(
        max_length=1000,
        widget=forms.Textarea(attrs={"rows": 1, "placeholder": "Ask a question about your PDFs"}),
    )


class HomeActionForm(forms.Form):
    action = forms.ChoiceField(
        choices=[("clear_answers", "Clear recent answers")],
    )


class DocumentActionForm(forms.Form):
    action = forms.ChoiceField(
        choices=[("reindex", "Re-index"), ("delete", "Delete")],
    )
