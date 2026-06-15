output "ec2_public_ip" {
  description = "Public IP of the MongoDB EC2 instance"
  value       = aws_instance.mongodb.public_ip
}

output "ec2_ssh_command" {
  description = "SSH command to connect to the MongoDB VM"
  value       = "ssh -i wiz-exercise/wiz-exercise-key ubuntu@${aws_instance.mongodb.public_ip}"
}

output "s3_bucket_name" {
  description = "S3 bucket for MongoDB backups"
  value       = aws_s3_bucket.db_backups.bucket
}

output "ecr_repository_url" {
  description = "ECR repository URL for pushing Docker images"
  value       = aws_ecr_repository.movie_buddy.repository_url
}

output "mongodb_url" {
  description = "MongoDB connection string for the app"
  value       = "mongodb://${var.mongodb_username}:${var.mongodb_password}@${aws_instance.mongodb.public_ip}:27017"
  sensitive   = true
}

output "eks_cluster_name" {
  description = "EKS cluster name"
  value       = aws_eks_cluster.main.name
}

output "eks_cluster_endpoint" {
  description = "EKS cluster API endpoint"
  value       = aws_eks_cluster.main.endpoint
}
# CI/CD test
