---
date: 2025-09-09
title: "FastAPI: Building Modern APIs with Python"
---

![fastAPILogo](https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png)

## Introduction
FastAPI is a modern, high-performance web framework for building APIs with Python. It leverages Python's type hints to provide automatic data validation, serialization, and interactive documentation. It is built on top of Starlette for the web parts and Pydantic for the data parts, ensuring that it is both fast and reliable. FastAPI is designed to be easy to use and learn, while also being highly performant, making it a popular choice for building web applications and APIs.

## What is FastAPI?
FastAPI is designed for speed and ease of use, making it ideal for developing RESTful APIs, web services, and even full web applications.

- **Key Features**:
    - **High Performance:** Comparable to Node.js and Go, thanks to Starlette and Pydantic.
    - **Automatic Documentation:** Generates interactive API docs with Swagger UI and ReDoc.
    - **Type Safety:** Uses Python type annotations for request/response validation.
    - **Asynchronous Support:** Built on ASGI for handling async operations.
    - **Dependency Injection:** Easy management of dependencies like databases or authentication.

- **Advantages**:
    - Reduces boilerplate code.  
    - Supports modern Python features (3.7+).  
    - Great for machine learning models, data APIs, and microservices.  

- **Considerations**:
    - Steeper learning curve if new to type hints or async programming.  
    - Best suited for API-focused projects rather than traditional web apps.

## Installation
To install FastAPI, you'll need Python 3.7 or higher. Use `pip3` for installation.

#### Basic Installation
```bash
pip3 install fastapi
```

#### Full Installation
This includes Uvicorn ASGI server for running the app:  
```bash
pip3 install fastapi[all]
```

#### Verification
Run a simple app to check:
```python
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Core Concepts
FastAPI revolves around a few key elements:

- **App Instance:** The central `FastAPI()` object where routes are defined.
- **Path Operations:** Decorators like `@app.get()`, `@app.post()` for HTTP methods.
- **Pydantic Models:** For data validation and serialization using Python classes.
- **Dependencies:** Functions or classes injected into path operations.
- **Middleware:** For handling requests globally (e.g., `CORS`).
- **Background Tasks:** For running non-blocking operations.

Example of a Pydantic model:

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None
```

## Request Handling
FastAPI handles incoming requests through path operations, allowing you to define request bodies using Pydantic models, simple GET and POST endpoints, and more. Below, we break it down into mini sections with examples.

### Simple GET Requests
Define a basic GET endpoint to return a simple response, useful for testing or lightweight data retrieval.

```python
# Simple GET endpoint
@app.get("/getreq")
def read_root():
    return {"Hello": "World"}
```

### Handling POST Requests with Parameters
Create a POST endpoint that accepts parameters (e.g., as query params or form data). This example uses simple parameters with defaults for flexibility.

```python
# Simple POST endpoint with parameters
@app.post("/postreq")
def create_item(name: str, price: int = 0):
    return {"item_received": {"name": name, "price": price}}
```

### Defining Pydantic Models for Input and Output
Use Pydantic to create models for validating request input and structuring response output. This ensures type safety and automatic serialization.

```python
from pydantic import BaseModel

# Pydantic model for request body as Input
class Item(BaseModel):
    name: str
    description: str | None = None
    price: int
    tax: float | None = None

# Pydantic model for response body as Output
class ItemResponse(BaseModel):
    name: str
    description: str | None = None
    price: int
    tax: float | None = None
    price_with_tax: float | None = None
```

### Handling POST Requests with JSON Body
Create a POST endpoint that accepts a JSON body via a Pydantic model, processes it (e.g., computes an additional field), and returns a response using another model.  
**Note:** GET requests cannot have a body, so POST is used here for body handling.

```python
from fastapi import FastAPI
# Assume Pydantic models are defined as above

app = FastAPI()

# POST endpoint that uses the Pydantic model. It takes a JSON body and returns it with an additional computed field.
@app.post("/pydanticreq", response_model=ItemResponse)
def create_item(item: Item):
    item_dict = item.model_dump()
    if item.tax is not None:
        item_dict["price_with_tax"] = item.price + item.tax
    return item_dict
```

### Full Example
Here's a complete example combining all the above concepts:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Pydantic model for request body as Input
class Item(BaseModel):
    name: str
    description: str | None = None
    price: int
    tax: float | None = None


# Pydantic model for response body as Output
class ItemResponse(BaseModel):
    name: str
    description: str | None = None
    price: int
    tax: float | None = None
    price_with_tax: float | None = None


# POST endpoint that uses the Pydantic model. It takes a JSON body and returns it with an additional computed field.
# NOTE: GET requests cannot have a body, so we use POST here.
@app.post("/pydanticreq", response_model=ItemResponse)
def create_item(item: Item):
    item_dict = item.model_dump()
    if item.tax is not None:
        item_dict["price_with_tax"] = item.price + item.tax
    return item_dict


# Simple GET and POST endpoints
@app.get("/getreq")
def read_root():
    return {"Hello": "World"}


# Simple POST endpoint with form data
@app.post("/postreq")
def create_item(name: str, price: int = 0):
    return {"item_received": {"name": name, "price": price}}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Handling Query and Routes
Endpoints are defined by routes, and query parameters allow dynamic data in URLs.

#### Path Parameters
Use curly braces to capture values from the URL.

```python
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

#### Query Parameters
These are automatically parsed from URL queries.

```python
@app.get("/items/")
async def read_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}
```

#### Optional and Default Values
These use type hints for validation.

```python
@app.get("/users/{user_id}")
async def read_user(user_id: str, q: str | None = None):
    return {"user_id": user_id, "query": q}
```

#### Route Grouping
These use APIRouter for modular routes.

```python
from fastapi import APIRouter

router = APIRouter()

@router.get("/users/")
async def read_users():
    return ["user1", "user2"]
```

## Handling Websockets
FastAPI enables real-time, two-way communication between clients and servers using WebSockets. This allows you to build interactive applications such as chat systems, live dashboards, and notifications.

To create a WebSocket endpoint, use the `@app.websocket()` decorator. The server can accept connections, receive messages from clients, and send responses back—all asynchronously.

```python
from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message text was: {data}")
```


## Handling Static Files
FastAPI makes it easy to serve static assets such as images, CSS, and JavaScript files. To do this, use the `StaticFiles` class and mount a directory to a specific route. This is useful for delivering frontend resources or public files directly from your application.

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
```

## Automatic Documentation
FastAPI auto-generates interactive docs based on OpenAPI standards. You can customize the API metadata like title, description, version, and tags to make the documentation more informative and organized.

#### Basic Access

- **Swagger UI:** Available at `/docs` for interactive testing.
- **ReDoc:** Available at `/redoc` for a clean, readable view.


#### Customizing App Metadata
These pass parameters to the FastAPI app instance to set title, description, version, terms of service, contact, and license info.

```python
from fastapi import FastAPI

app = FastAPI(
    title="My FastAPI App",
    description="A sample API for managing items and users.",
    version="1.0.0",
    terms_of_service="http://example.com/terms/",
    contact={
        "name": "API Support",
        "url": "http://www.example.com/support",
        "email": "support@example.com"
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html"
    }
)
```

#### Adding Tags
These use `openapi_tags` to define tag metadata for grouping endpoints in docs. Tags help organize routes logically.

```python
tags_metadata = [
    {
        "name": "items",
        "description": "Operations related to items."
    },
    {
        "name": "users",
        "description": "Manage users and authentication."
    }
]

app = FastAPI(openapi_tags=tags_metadata)
```

#### Tagging Routes
Here assign tags to individual path operations for categorization in Swagger/ReDoc.

```python
@app.get("/items/", tags=["items"])
async def read_items():
    return [{"item_id": "Foo"}]

@app.get("/users/", tags=["users"])
async def read_users():
    return [{"username": "Rick"}]
```

#### Route-Specific Details 
Here you can add summary and description to endpoints for better clarity in docs.

```python
@app.post("/items/", tags=["items"], summary="Create an item", description="Create a new item with name, price, and optional tax.")
async def create_item(item: Item):
    return item
```

#### URL Customization
Here you can also change or disable doc URLs.

```python
app = FastAPI(docs_url="/api-docs", redoc_url="/api-redoc")  # Custom paths
app = FastAPI(docs_url=None, redoc_url=None)  # Disable docs
```

## Conclusion
FastAPI stands out as a modern, efficient, and developer-friendly framework for building APIs and web services with Python. Its emphasis on type safety, automatic documentation, and asynchronous support makes it a powerful tool for both rapid prototyping and production-grade applications. By leveraging Pydantic and Starlette, FastAPI ensures robust data validation and high performance, while its intuitive design reduces boilerplate and streamlines development.

Whether you are creating simple endpoints, handling complex data models, integrating real-time features with WebSockets, or serving static files, FastAPI provides clear patterns and built-in solutions. The automatic generation of interactive documentation further enhances the developer experience, making it easy to test and share your APIs.

In summary, FastAPI is an excellent choice for anyone looking to build scalable, maintainable, and well-documented APIs in Python. Its combination of speed, reliability, and ease of use continues to drive its popularity in the Python ecosystem.

## References
- [Official FastAPI Documentation](https://fastapi.tiangolo.com/)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [GitHub Repository](https://github.com/tiangolo/fastapi)
- [Real-World Examples](https://fastapi.tiangolo.com/advanced/)
