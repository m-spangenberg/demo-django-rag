from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="ChatExchange",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("session_key", models.CharField(max_length=40)),
                ("question", models.TextField()),
                ("answer", models.TextField(blank=True)),
                ("sources", models.JSONField(blank=True, default=list)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={"ordering": ["-created_at"]},
        ),
        migrations.CreateModel(
            name="Document",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("title", models.CharField(max_length=255)),
                ("source_file", models.FileField(upload_to="documents/")),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("pending", "Pending"),
                            ("processing", "Processing"),
                            ("ready", "Ready"),
                            ("failed", "Failed"),
                        ],
                        default="pending",
                        max_length=20,
                    ),
                ),
                ("chunk_count", models.PositiveIntegerField(default=0)),
                ("page_count", models.PositiveIntegerField(default=0)),
                ("pinecone_namespace", models.CharField(blank=True, max_length=255)),
                ("error_message", models.TextField(blank=True)),
                ("uploaded_at", models.DateTimeField(auto_now_add=True)),
                ("processed_at", models.DateTimeField(blank=True, null=True)),
            ],
            options={"ordering": ["-uploaded_at"]},
        ),
        migrations.CreateModel(
            name="DocumentChunk",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("chunk_index", models.PositiveIntegerField()),
                ("page_number", models.PositiveIntegerField(default=1)),
                ("vector_id", models.CharField(max_length=255, unique=True)),
                ("text_preview", models.TextField()),
                ("metadata", models.JSONField(blank=True, default=dict)),
                (
                    "document",
                    models.ForeignKey(on_delete=models.deletion.CASCADE, related_name="chunks", to="rag.document"),
                ),
            ],
            options={"ordering": ["document", "chunk_index"], "unique_together": {("document", "chunk_index")}},
        ),
    ]
