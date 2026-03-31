from __future__ import annotations

from django.db import models
from django.utils import timezone


class Document(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        PROCESSING = "processing", "Processing"
        READY = "ready", "Ready"
        FAILED = "failed", "Failed"

    title = models.CharField(max_length=255)
    source_file = models.FileField(upload_to="documents/")
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING,
    )
    chunk_count = models.PositiveIntegerField(default=0)
    page_count = models.PositiveIntegerField(default=0)
    pinecone_namespace = models.CharField(max_length=255, blank=True)
    error_message = models.TextField(blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def mark_processing(self) -> None:
        self.status = self.Status.PROCESSING
        self.error_message = ""
        self.save(update_fields=["status", "error_message"])

    def mark_ready(self, *, chunk_count: int, page_count: int, namespace: str) -> None:
        self.status = self.Status.READY
        self.chunk_count = chunk_count
        self.page_count = page_count
        self.pinecone_namespace = namespace
        self.processed_at = timezone.now()
        self.error_message = ""
        self.save(
            update_fields=[
                "status",
                "chunk_count",
                "page_count",
                "pinecone_namespace",
                "processed_at",
                "error_message",
            ]
        )

    def mark_failed(self, message: str) -> None:
        self.status = self.Status.FAILED
        self.error_message = message
        self.save(update_fields=["status", "error_message"])

    def __str__(self) -> str:
        return self.title


class DocumentChunk(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="chunks")
    chunk_index = models.PositiveIntegerField()
    page_number = models.PositiveIntegerField(default=1)
    vector_id = models.CharField(max_length=255, unique=True)
    text_preview = models.TextField()
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["document", "chunk_index"]
        unique_together = ("document", "chunk_index")

    def __str__(self) -> str:
        return f"{self.document.pk}:{self.chunk_index}"


class ChatExchange(models.Model):
    session_key = models.CharField(max_length=40)
    question = models.TextField()
    answer = models.TextField(blank=True)
    sources = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    @property
    def question_preview(self) -> str:
        return self.question[:60]

    def __str__(self) -> str:
        return self.question_preview
