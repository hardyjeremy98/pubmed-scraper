import requests


class HTTPSession:
    """Manages HTTP session with proper headers."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )

    def get(self, url: str, **kwargs) -> requests.Response:
        """Make GET request using the session."""
        return self.session.get(url, **kwargs)
