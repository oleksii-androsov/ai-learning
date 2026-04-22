output "public_ip" {
  description = "Public IP of the new instance — update EC2_HOST with this value"
  value       = aws_instance.rag.public_ip
}

output "ssh_command" {
  description = "SSH into the instance"
  value       = "ssh -i your-key.pem ubuntu@${aws_instance.rag.public_ip}"
}

output "api_url" {
  description = "RAG API base URL"
  value       = "http://${aws_instance.rag.public_ip}:8000"
}

output "streamlit_url" {
  description = "Streamlit UI"
  value       = "http://${aws_instance.rag.public_ip}:8501"
}

output "update_github_secret" {
  description = "Run this after apply to point CI/CD at the new instance"
  value       = "gh secret set EC2_HOST --body ${aws_instance.rag.public_ip}"
}
