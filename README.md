## Bree-MD Code Directory
``` markdown
src/
├── core/
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py        # LLM client implementations
│   │   ├── agent.py         # Agent management
│   │   └── prompter.py      # Prompt engineering
│   ├── agents/
│   │   ├── base.py          # Base agent class
│   │   └── registry.py      # Agent registry
│   ├── tools/
│   │   ├── base.py          # Base tool interface
│   │   └── registry.py      # Tool registry
│   ├── schemas/
│   │   └── models.py        # Pydantic models
├── config/
│   ├── __init__.py
│   ├── settings.py          # App configuration
│   └── security.py          # Security settings
├── utils/
│   ├── error_handling.py
│   ├── sanitization.py
│   └── logging.py
└── main.py                  # Execution entry point

```