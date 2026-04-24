from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run("assistant_conflict_env.server:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
