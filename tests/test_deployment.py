import unittest
from pathlib import Path


class DockerDeploymentTests(unittest.TestCase):
    def test_dockerignore_excludes_local_env_files_but_keeps_example(self) -> None:
        dockerignore = Path(".dockerignore").read_text(encoding="utf-8").splitlines()

        self.assertIn(".env", dockerignore)
        self.assertIn(".env.*", dockerignore)
        self.assertIn("!.env.example", dockerignore)
        self.assertIn("*.pem", dockerignore)
        self.assertIn("*.key", dockerignore)
        self.assertIn("*.crt", dockerignore)
        self.assertIn("*.p12", dockerignore)


if __name__ == "__main__":
    unittest.main()
