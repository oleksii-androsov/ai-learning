resource "aws_s3_bucket" "db_backups" {
  bucket        = "${var.project_name}-db-backups"
  force_destroy = true

  tags = { Name = "${var.project_name}-db-backups" }
}

# Intentional weakness: disable the block that prevents public access
resource "aws_s3_bucket_public_access_block" "db_backups" {
  bucket = aws_s3_bucket.db_backups.id

  block_public_acls       = false
  block_public_policy     = false
  ignore_public_acls      = false
  restrict_public_buckets = false
}

# Intentional weakness: allow anyone to read and list the bucket
resource "aws_s3_bucket_policy" "db_backups" {
  bucket = aws_s3_bucket.db_backups.id

  depends_on = [aws_s3_bucket_public_access_block.db_backups]

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "PublicReadAndList"
        Effect    = "Allow"
        Principal = "*"
        Action    = ["s3:GetObject", "s3:ListBucket"]
        Resource  = [
          aws_s3_bucket.db_backups.arn,
          "${aws_s3_bucket.db_backups.arn}/*"
        ]
      }
    ]
  })
}
