from django import forms


class UploadDocumentForm(forms.Form):
    title = forms.CharField(max_length=255)
    pdf = forms.FileField(help_text="Upload a PDF file for ingestion.")

    def clean_pdf(self):
        pdf = self.cleaned_data["pdf"]
        if not pdf.name.lower().endswith(".pdf"):
            raise forms.ValidationError("Only PDF files are supported.")
        return pdf


class ChatQueryForm(forms.Form):
    question = forms.CharField(
        max_length=1000,
        widget=forms.Textarea(attrs={"rows": 3, "placeholder": "Ask a question about your PDFs"}),
    )


class DocumentActionForm(forms.Form):
    action = forms.ChoiceField(
        choices=[("reindex", "Re-index"), ("delete", "Delete")],
    )
