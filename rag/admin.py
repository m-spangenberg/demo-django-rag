from django.contrib import admin

from .models import ChatExchange, Document, DocumentChunk


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = (
        "title",
        "status",
        "chunk_count",
        "uploaded_at",
        "processed_at",
    )
    list_filter = ("status", "uploaded_at")
    search_fields = ("title", "source_file")


@admin.register(DocumentChunk)
class DocumentChunkAdmin(admin.ModelAdmin):
    list_display = ("document", "chunk_index", "page_number", "vector_id")
    list_select_related = ("document",)
    search_fields = ("document__title", "text_preview", "vector_id")


@admin.register(ChatExchange)
class ChatExchangeAdmin(admin.ModelAdmin):
    list_display = ("created_at", "session_key", "question_preview")
    readonly_fields = ("created_at",)
    search_fields = ("session_key", "question", "answer")
