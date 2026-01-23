# Session 13: Model Context Protocol (MCP) - First Principles System Design

**Target Audience:** Principal Engineers (12+ years) transitioning to Generative AI  
**Philosophy:** Treat AI as a probabilistic software component, not magic  
**Approach:** Backend engineering primitives applied to AI infrastructure

---

## Table of Contents

1. **The Landscape: Pre-MCP Tool Calling Chaos**
2. **The Core Primitive: Deconstructing MCP**
3. **Problem Statement: Why Standardization Matters**
4. **Architecture: Host, Client, Server, Transport**
5. **Transport Mechanisms: stdio vs SSE**
6. **Tool Lifecycle: Discovery, Invocation, Execution**
7. **Implementation: Building Your First MCP Server**
8. **Integration: Connecting to AI Applications**
9. **Security & Authentication Patterns**
10. **Observability & Control Flow**
11. **Role-Specific Scenarios - Part 1: Backend/Cloud & DevOps**
12. **Role-Specific Scenarios - Part 2: SRE, Platform, Cloud Leader**
13. **Practical Exercise: Custom MCP Implementation**

---

## Learning Outcomes

By the end of this session, you will:
- Understand MCP as a **protocol specification**, not a framework
- Design scalable tool architectures for AI systems using familiar patterns
- Implement custom MCP servers with proper separation of concerns
- Make informed architectural decisions for AI infrastructure
- Debug and observe MCP interactions in production systems

---

## Step 1: The Landscape - Pre-MCP Tool Calling Chaos

### The Fragmentation Problem

Before MCP, every AI application developer solved the same problem differently: **How do I connect my LLM to external capabilities?**

Think back to the pre-REST era of web services. Remember SOAP, XML-RPC, custom binary protocols? Every vendor had their own approach. The same fragmentation existed in AI tool calling.

### The Custom Implementation Pattern

Let's deconstruct a typical pre-MCP implementation:

**Scenario:** You're building an AI assistant that needs to send emails via Gmail.

```python
# Your custom tool definition (circa 2023)
SYSTEM_PROMPT = """
You have access to the following tools:

send_email:
  - Description: Send an email via Gmail
  - Parameters: to (string), subject (string), body (string)
  
To use a tool, respond with:
ACTION: tool_name
PARAMS: {"param1": "value1", "param2": "value2"}

I will then execute the tool and return the result.
"""

# Your custom tool registry (manual mapping)
TOOL_MAP = {
    "send_email": send_email_function,
    "search_web": search_web_function,
    "query_database": query_db_function
}

# Your custom execution loop
def execute_tool(tool_name, params):
    if tool_name in TOOL_MAP:
        return TOOL_MAP[tool_name](**params)
    else:
        return {"error": "Tool not found"}
```

**Backend Engineering Analogy:** This is like every developer writing their own HTTP server from scratch. You're reinventing:
- Request routing (which function to call?)
- Parameter validation (is the input valid?)
- Error handling (what if the tool fails?)
- Response formatting (how to return results?)

### The OpenAI Function Calling "Improvement"

OpenAI introduced structured function calling in 2023. This was better:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email via Gmail",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"}
                },
                "required": ["to", "subject", "body"]
            }
        }
    }
]

# OpenAI tells you WHICH tool to call and WITH WHAT params
response = openai.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=tools
)

# But YOU still have to:
# 1. Maintain the tool registry
# 2. Map tool names to actual functions
# 3. Execute the function
# 4. Format the result
# 5. Send it back to the LLM
```

**The Problem:** The tool definition is standardized, but:
- The **execution layer** is still custom
- Tools are **not portable** across different AI applications
- Each developer maintains their own tool registry
- Google, Microsoft, Anthropic all have slightly different schemas

### The Shareability Problem

**Real-World Scenario:** Google wants to provide Gmail tools for AI applications.

In the pre-MCP world:
- Google cannot provide a "ready-to-use" tool package
- Every developer must:
  - Read Google's API documentation
  - Write their own wrapper function
  - Define the tool schema in their app's format
  - Handle authentication and rate limiting

**Backend Engineering Analogy:** This is like every developer writing their own OAuth client library. There's no standardized SDK.

### Why This Matters at Scale

Consider an enterprise with:
- 50+ microservices
- 10+ databases (Postgres, MongoDB, Snowflake, etc.)
- 20+ SaaS tools (Gmail, Slack, Jira, etc.)
- Custom internal APIs

Without standardization:
- Each AI application developer writes custom integrations (100s of man-hours)
- Tools are not reusable across teams
- Maintenance nightmare (API changes, security patches)
- No ecosystem effects (no marketplace, no community contributions)

**The Market Signal:** When Google, OpenAI, Anthropic, and others all adopt the same protocol, it's not hype—it's recognition that the problem is real and the solution is sound.

---

## Step 2: The Core Primitive - Deconstructing MCP

### Breaking Down "Model Context Protocol"

Let's parse this like a system design interview:

**Model:** The LLM (GPT-4, Claude, Gemini, etc.)  
**Context:** External data, tools, capabilities, resources  
**Protocol:** A standardized set of rules for communication

**MCP Definition (Engineering Perspective):**
> A client-server protocol that standardizes how AI applications discover, invoke, and consume external tools and data sources.

### The "USB-C for AI" Analogy

Remember when every device had a different charging port? MicroUSB, MiniUSB, Lightning, proprietary connectors?

**Pre-USB-C World (Tool Calling):**
- Device manufacturer (Google) creates a custom connector
- You (developer) need a custom cable/adapter for each device
- Not interchangeable or portable

**Post-USB-C World (MCP):**
- Standardized interface on both sides
- Any USB-C device works with any USB-C cable
- Manufacturers compete on implementation, not interface

### Protocol vs Implementation

This is crucial to understand:

**MCP is a PROTOCOL, not a library or framework.**

```
┌─────────────────────────────────────────────────────┐
│                    PROTOCOL LAYER                    │
│  (Defines WHAT messages look like, not HOW to send) │
├─────────────────────────────────────────────────────┤
│                TRANSPORT LAYER (HOW)                 │
│         stdio, SSE, WebSockets (future)             │
├─────────────────────────────────────────────────────┤
│              IMPLEMENTATION LAYER (CODE)             │
│     Python SDK, TypeScript SDK, Java SDK, etc.      │
└─────────────────────────────────────────────────────┘
```

**Backend Engineering Analogy:** Think of HTTP.

- **HTTP** is a protocol (defines request/response format)
- **TCP** is the transport (how bytes move)
- **Express.js, FastAPI, Spring Boot** are implementations

You can implement HTTP over different transports (TCP, QUIC) and in different languages (Python, Go, Rust).

### The OSI Model Parallel

For distributed systems engineers, here's the mental model:

```
┌──────────────────────┬──────────────────────────────┐
│   OSI Layer          │   MCP Equivalent             │
├──────────────────────┼──────────────────────────────┤
│ Application Layer    │ Tool Logic (send_email())    │
│ Presentation Layer   │ JSON-RPC Message Format      │
│ Session Layer        │ MCP Protocol (tool discovery)│
│ Transport Layer      │ stdio / SSE / WebSockets     │
│ Network Layer        │ TCP/IP (if remote)           │
└──────────────────────┴──────────────────────────────┘
```

### What MCP Actually Standardizes

**1. Message Format (JSON-RPC 2.0)**

```json
// Tool discovery request
{
  "jsonrpc": "2.0",
  "method": "tools/list",
  "id": 1
}

// Tool discovery response
{
  "jsonrpc": "2.0",
  "result": {
    "tools": [
      {
        "name": "send_email",
        "description": "Send an email via Gmail",
        "inputSchema": {
          "type": "object",
          "properties": {
            "to": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"}
          },
          "required": ["to", "subject", "body"]
        }
      }
    ]
  },
  "id": 1
}
```

**2. Tool Invocation Flow**

```json
// Tool execution request
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "send_email",
    "arguments": {
      "to": "user@example.com",
      "subject": "Test",
      "body": "Hello World"
    }
  },
  "id": 2
}

// Tool execution response
{
  "jsonrpc": "2.0",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Email sent successfully"
      }
    ]
  },
  "id": 2
}
```

**Backend Engineering Analogy:** This is like OpenAPI/Swagger spec. It defines:
- Endpoint structure
- Request/response schemas
- Data types and validation rules

But unlike REST APIs, MCP is:
- **Stateful** (persistent connection)
- **Bidirectional** (server can push updates)
- **Dynamic** (tools can be discovered at runtime)

### The Key Insight: Separation of Concerns

```
┌─────────────────────────────────────────────────────────┐
│              YOUR AI APPLICATION (Host)                  │
│  - Manages conversation state                           │
│  - Decides when to call tools                           │
│  - Presents results to user                             │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ MCP Client (handles protocol)
                     │
┌────────────────────▼────────────────────────────────────┐
│              MCP SERVER (Tool Provider)                  │
│  - Registers available tools                            │
│  - Validates input parameters                           │
│  - Executes tool logic                                  │
│  - Returns formatted results                            │
└─────────────────────────────────────────────────────────┘
```

**What You Own as a Developer:**
- Tool implementation (business logic)
- Authentication/authorization (API keys, OAuth)
- Error handling and retries
- Logging and observability

**What MCP Handles:**
- Tool discovery mechanism
- Message serialization/deserialization
- Request routing
- Result formatting

### Why "Protocol" Matters

Protocols enable **ecosystem effects**:

1. **Interoperability:** Any MCP client can talk to any MCP server
2. **Marketplace:** Third parties can publish MCP servers (like npm packages)
3. **Tooling:** IDE extensions, debuggers, observability tools
4. **Competition:** Multiple implementations can coexist

**Backend Engineering Analogy:** Think about the HTTP protocol ecosystem:
- Browsers (clients) work with any web server
- Libraries (requests, axios, fetch) abstract the protocol
- Proxies, load balancers, CDNs emerge as middleware
- Observability tools (Wireshark, browser DevTools) work universally

### The Non-Technical Reason for Adoption

**Network Effects:** When Anthropic (Claude) released MCP, they could have kept it proprietary. But by making it open and getting buy-in from OpenAI, Google, and Microsoft:

- Developers build once, deploy everywhere
- Tool providers (Google, Slack, etc.) build once, support all AI apps
- Community contributions accelerate (GitHub: 1000+ MCP servers in 3 months)

**This is not hype. This is engineering pragmatism.**

---

### Checkpoint Question 1: Architecture Design

**FAANG-Style Question:**

You're the principal engineer at a B2B SaaS company with 200+ enterprise customers. Your product is a CRM platform with 30+ internal microservices (user management, billing, analytics, email campaigns, etc.).

The CEO wants to add an "AI Assistant" feature that can:
- Answer questions about customer data (queries 5 databases: Postgres, MongoDB, Redis, Snowflake, Elasticsearch)
- Perform actions (create tickets, send emails, update records)
- Integrate with customer's own tools (Slack, Jira, Salesforce, custom webhooks)

**Design Question:**
1. Would you use MCP for this architecture? Why or why not?
2. If yes, where would you place MCP servers in your system?
3. How would you handle authentication and rate limiting?
4. What are the latency, security, and cost trade-offs?

**Expected Answer:**

```mermaid
sequenceDiagram
    participant User
    participant AIApp as AI Application<br/>(MCP Host)
    participant MCPGateway as MCP Gateway<br/>(Orchestrator)
    participant MCPServer1 as MCP Server<br/>Internal Services
    participant MCPServer2 as MCP Server<br/>Customer Tools
    participant InternalDB as Internal DBs
    participant External as External APIs

    User->>AIApp: "Create a ticket for Acme Corp"
    AIApp->>MCPGateway: Discover available tools
    MCPGateway->>MCPServer1: List internal tools
    MCPServer1-->>MCPGateway: [create_ticket, query_customer, etc.]
    MCPGateway->>MCPServer2: List customer tools
    MCPServer2-->>MCPGateway: [slack_notify, jira_create, etc.]
    MCPGateway-->>AIApp: Aggregated tool list
    AIApp->>AIApp: LLM decides to use create_ticket
    AIApp->>MCPGateway: Execute create_ticket(customer_id, title, body)
    MCPGateway->>MCPServer1: Forward request + auth context
    MCPServer1->>InternalDB: Validate permissions & create ticket
    InternalDB-->>MCPServer1: ticket_id: 12345
    MCPServer1-->>MCPGateway: Success response
    MCPGateway-->>AIApp: Tool result
    AIApp-->>User: "Ticket #12345 created successfully"
```

**Detailed Answer:**

**1. Should you use MCP?**

**YES**, for these reasons:

- **Separation of Concerns:** Your AI application logic (conversation, context management) is separate from tool implementations
- **Team Scalability:** Each microservice team can own their MCP server independently
- **Customer Extensibility:** Customers can add their own MCP servers without you modifying your core AI app
- **Protocol Stability:** As you iterate on AI models (GPT-4 → GPT-5 → Claude 4), tool integrations don't break

**2. MCP Server Placement:**

**Option A: Single MCP Server (Anti-Pattern for Scale)**
```
❌ Monolithic MCP Server
   ├─ All 30+ microservices in one server
   ├─ Single point of failure
   └─ Tight coupling
```

**Option B: One MCP Server per Domain (Recommended)**
```
✅ Domain-Driven MCP Servers
   ├─ Customer MCP Server (Postgres queries, customer CRUD)
   ├─ Billing MCP Server (Stripe integration, invoice generation)
   ├─ Communication MCP Server (email, SMS, Slack)
   ├─ Analytics MCP Server (Snowflake queries, report generation)
   └─ Customer-Provided MCP Server (their own tools)
```

**Why Domain-Driven?**
- **Fault Isolation:** Billing server crash doesn't break customer queries
- **Independent Scaling:** Scale analytics server separately (heavy queries)
- **Team Ownership:** Billing team owns billing MCP server
- **Rate Limiting:** Apply per-domain limits

**3. Authentication & Rate Limiting:**

```typescript
// MCP Gateway (API Gateway pattern)
class MCPGateway {
  async executeToolWithAuth(
    userId: string,
    customerId: string,
    toolName: string,
    params: any
  ) {
    // 1. Authentication
    const authContext = await this.validateUserPermissions(userId, customerId);
    
    // 2. Rate Limiting (per customer, per tool)
    const rateLimitKey = `${customerId}:${toolName}`;
    if (await this.rateLimiter.isRateLimited(rateLimitKey)) {
      throw new RateLimitError("Too many requests");
    }
    
    // 3. Route to appropriate MCP server
    const mcpServer = this.routeTool(toolName);
    
    // 4. Execute with auth context
    const result = await mcpServer.execute(toolName, params, authContext);
    
    // 5. Audit logging
    await this.auditLog.record({
      userId,
      customerId,
      toolName,
      params: sanitizeForLogs(params),
      result: sanitizeForLogs(result),
      timestamp: Date.now()
    });
    
    return result;
  }
}
```

**Rate Limiting Strategy:**
- **Per-Customer Limits:** 1000 tool calls/hour per customer (prevent abuse)
- **Per-Tool Limits:** Heavy operations (Snowflake queries) have lower limits
- **Burst Allowance:** Allow short bursts for good UX, but enforce hourly quotas

**4. Trade-offs Analysis:**

| Aspect | Without MCP | With MCP |
|--------|-------------|----------|
| **Latency** | Direct function calls (~10ms) | Protocol overhead (~20-50ms per tool call) |
| **Scalability** | Tightly coupled, hard to scale teams | Each team scales independently |
| **Security** | Custom auth per integration | Centralized auth gateway pattern |
| **Extensibility** | Requires code changes for new tools | Customer can add their own servers |
| **Debuggability** | Custom logging per integration | Standardized observability tooling |
| **Cost** | Lower compute (no protocol overhead) | Slightly higher compute, but lower engineering cost |

**When NOT to use MCP:**
- Ultra-low-latency requirements (<10ms per tool call)
- Simple applications with 2-3 hardcoded tools
- Internal-only tools with no extensibility needs
- Embedded systems with extreme resource constraints

**Key Insight:** The 20-50ms protocol overhead is negligible compared to:
- LLM inference time (500-2000ms)
- External API calls (100-500ms)
- Database queries (10-100ms)

**Cost-Benefit:** The engineering velocity gain (teams work independently, customers can extend) far outweighs the minor latency cost.

---

## Step 3: Problem Statement - Why Standardization Matters

### The Market Forces Behind MCP

Let's think about this from first principles. Why do protocols succeed or fail?

**Successful Protocols:**
- HTTP/REST (replaced SOAP, XML-RPC)
- OAuth 2.0 (replaced proprietary auth schemes)
- gRPC (standardized RPC across languages)
- OpenTelemetry (unified observability)

**Failed Protocols:**
- CORBA (too complex)
- SOAP (verbose, rigid)
- GraphQL Federation v1 (fragmentation)

**Common Success Factors:**
1. **Solves a real pain point** (not theoretical)
2. **Simple enough to adopt quickly** (<1 day to first integration)
3. **Backed by multiple large players** (prevents lock-in fears)
4. **Extensible without breaking** (protocol versioning)

MCP checks all these boxes. But let's dig deeper into the **specific** problems it solves.

### Problem 1: The "N × M Integration Matrix"

**Scenario:** You have:
- N = 5 AI applications (Claude Desktop, Cursor, Copilot, custom internal app, customer-facing chatbot)
- M = 20 data sources/tools (Postgres, Slack, Gmail, Jira, Snowflake, etc.)

**Without standardization:**
- Each AI app needs custom integration for each tool
- Total integrations: **N × M = 100 custom integrations**
- Each integration has custom code, auth, error handling
- Maintenance cost: 100 codebases to update when APIs change

**With MCP standardization:**
- Write M = 20 MCP servers (one per tool)
- Each AI app implements 1 MCP client
- Total integrations: **N + M = 25 components**
- Maintenance cost: Update MCP server once, all AI apps benefit

**Backend Engineering Analogy:** This is the "database driver" problem.

```
Without standardization:
┌─────────┐   ┌─────────┐   ┌─────────┐
│  App 1  │   │  App 2  │   │  App 3  │
└────┬────┘   └────┬────┘   └────┬────┘
     │             │             │
     ├─────────────┼─────────────┤
     │             │             │
┌────▼────┐   ┌───▼────┐   ┌───▼────┐
│Custom DB│   │Custom DB│  │Custom DB│
│ Driver  │   │ Driver  │  │ Driver  │
└────┬────┘   └────┬────┘   └────┬────┘
     │             │             │
┌────▼────────────▼─────────────▼────┐
│          Database                   │
└─────────────────────────────────────┘

With standardization (JDBC, SQLAlchemy, etc.):
┌─────────┐   ┌─────────┐   ┌─────────┐
│  App 1  │   │  App 2  │   │  App 3  │
│ (JDBC)  │   │ (JDBC)  │   │ (JDBC)  │
└────┬────┘   └────┬────┘   └────┬────┘
     │             │             │
     └─────────────┼─────────────┘
                   │
            ┌──────▼──────┐
            │ DB Driver   │
            │ (Standard)  │
            └──────┬──────┘
                   │
            ┌──────▼──────┐
            │  Database   │
            └─────────────┘
```

### Problem 2: Tool Portability and Reusability

**Real-World Example:** Slack wants to provide AI integrations.

**Pre-MCP World:**
```python
# Slack must provide separate SDKs for each AI platform

# For OpenAI function calling
slack_openai_tools = [
    {
        "type": "function",
        "function": {
            "name": "send_slack_message",
            "description": "Send a message to Slack",
            "parameters": {...}
        }
    }
]

# For Anthropic
slack_anthropic_tools = [
    {
        "name": "send_slack_message",
        "description": "Send a message to Slack",
        "input_schema": {...}  # Different format!
    }
]

# For Google Gemini
slack_gemini_tools = [...]  # Yet another format!

# Plus: Each platform has different execution patterns
# - OpenAI: You call the function yourself
# - Anthropic: Different response format
# - Gemini: Different parameter passing
```

**MCP World:**
```bash
# Slack publishes ONE MCP server
npm install @slack/mcp-server

# Works with ANY MCP-compatible AI application
# - Cursor
# - Claude Desktop
# - Your custom app
# - Future applications you haven't built yet
```

**The Ecosystem Effect:**
- Slack writes integration once
- All AI apps benefit
- Community can contribute improvements
- Third parties can build extensions (like "Slack + Jira" orchestration)

### Problem 3: Security and Credential Management

**Scenario:** Your AI assistant needs access to:
- Production Postgres database (read-only)
- Snowflake data warehouse (analyst role)
- Gmail API (send on behalf of user)
- Internal microservices (service-to-service auth)

**Without standardization:**
```python
# Each tool has custom auth handling
def query_postgres(query):
    conn = psycopg2.connect(
        host=os.getenv("PG_HOST"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD")  # Stored where?
    )
    # Custom connection pooling, retry logic, etc.

def send_gmail(to, subject, body):
    creds = get_gmail_oauth_token()  # Custom OAuth flow
    service = build('gmail', 'v1', credentials=creds)
    # Custom error handling, rate limiting, etc.

# Every tool reimplements:
# - Credential storage
# - Token refresh
# - Connection pooling
# - Error handling
# - Audit logging
```

**With MCP (Separation of Concerns):**
```python
# MCP Server handles tool logic, NOT credential management
class PostgresMCPServer:
    def __init__(self, connection_pool):
        self.pool = connection_pool  # Injected by host
    
    @tool
    def query_database(self, query: str) -> str:
        # Only business logic here
        with self.pool.get_connection() as conn:
            result = conn.execute(query)
            return result.to_json()

# MCP Host (AI application) handles credentials
class AIApplicationHost:
    def __init__(self):
        self.credential_manager = VaultCredentialManager()
        self.postgres_server = PostgresMCPServer(
            connection_pool=self.create_postgres_pool()
        )
    
    def create_postgres_pool(self):
        creds = self.credential_manager.get_credentials("postgres")
        return ConnectionPool(
            host=creds.host,
            user=creds.user,
            password=creds.password,
            max_connections=10,
            idle_timeout=300
        )
```

**Key Architectural Insight:**
- **MCP Server:** "Thin wrapper" with zero hardcoded credentials
- **MCP Host:** Manages authentication, secrets, connection pooling
- **Benefit:** MCP servers can be open-sourced safely (no secrets)

**Backend Engineering Analogy:** This is like the distinction between:
- **Database Driver** (knows how to execute queries)
- **Connection String** (credentials, managed by application)

The driver doesn't store credentials; it's passed at runtime.

### Problem 4: Observability and Debugging

**Without standardization:**
```python
# Every tool has custom logging
def send_email(to, subject, body):
    logger.info(f"Sending email to {to}")  # Custom format
    try:
        result = gmail.send(...)
        logger.info(f"Email sent successfully")  # Custom format
        return result
    except Exception as e:
        logger.error(f"Failed to send email: {e}")  # Custom format
        raise

# No standardized way to:
# - Trace a request across multiple tools
# - Measure tool latency
# - Detect errors vs successes
# - Replay failed requests
```

**With MCP (Standardized Observability):**
```typescript
// MCP Gateway with OpenTelemetry integration
class MCPGateway {
  async executeToolWithTracing(toolName: string, params: any) {
    const span = tracer.startSpan('mcp.tool.execute', {
      attributes: {
        'mcp.tool.name': toolName,
        'mcp.server.id': this.getServerId(toolName),
        'mcp.params.count': Object.keys(params).length
      }
    });
    
    try {
      const result = await this.executeToolInternal(toolName, params);
      
      span.setAttributes({
        'mcp.result.size': JSON.stringify(result).length,
        'mcp.status': 'success'
      });
      
      return result;
    } catch (error) {
      span.recordException(error);
      span.setStatus({ code: SpanStatusCode.ERROR });
      throw error;
    } finally {
      span.end();
    }
  }
}
```

**Observability Benefits:**
- **Distributed Tracing:** See entire request flow (LLM → MCP Gateway → Tool → Database)
- **Standardized Metrics:** Tool call latency, success rate, error types
- **Protocol Inspection:** Debug tools can parse MCP messages (like Wireshark for HTTP)

### Problem 5: Version Management and Evolution

**Scenario:** OpenAI releases a new model that supports parallel tool calling.

**Without standardization:**
```python
# You must update every tool integration manually
# Old code:
result = execute_tool(tool_name, params)

# New code (parallel):
results = execute_tools_parallel([
    (tool_name_1, params_1),
    (tool_name_2, params_2)
])

# Every integration point must be updated
# Risk of breaking changes across 100s of integrations
```

**With MCP:**
```typescript
// MCP protocol evolves with backward compatibility
// MCP v1.0: Sequential tool calls
// MCP v1.1: Parallel tool calls (new capability)

// Old MCP servers work unchanged
// New MCP servers can opt-in to parallel execution
class ModernMCPServer {
  capabilities = {
    parallel_execution: true,  // New capability
    streaming: true,           // Future capability
    batch_operations: false    // Not implemented yet
  };
  
  // Protocol negotiation at connection time
  async connect(client) {
    const capabilities = await client.getCapabilities();
    if (capabilities.parallel_execution && this.supportsParallel) {
      this.enableParallelMode();
    }
  }
}
```

**Backend Engineering Analogy:** This is like HTTP/2 backward compatibility with HTTP/1.1.
- Servers advertise capabilities
- Clients negotiate features
- Both can work together even at different versions

### The Strategic Question: Build vs Adopt

**As a Principal Engineer, you're asked:**
> "Should we build our own tool calling protocol or adopt MCP?"

**Build Your Own (Red Flags):**
- ✓ You have unique requirements no one else has
- ✓ You can invest 10+ engineer-years building ecosystem
- ✓ You can convince third parties to support your protocol
- ✗ You want "control" (NIH syndrome)
- ✗ "It's not that complex, we can build it in a sprint" (famous last words)

**Adopt MCP (Green Flags):**
- ✓ Your requirements are common (90% of AI applications)
- ✓ You want ecosystem effects (community tools)
- ✓ You need to move fast (time to market)
- ✓ You want to attract talent (developers know MCP)
- ✓ You want third-party integrations (Slack, Google, etc.)

**The Reality:** Unless you're Google/OpenAI/Anthropic with resources to build an entire ecosystem, **adopt the standard**.

---

## Step 4: Architecture - Host, Client, Server, Transport

### The Core Components

Let's build the mental model from the ground up. MCP has **four** distinct components:

```
┌─────────────────────────────────────────────────────────────┐
│                     MCP Host                                │
│  (Your AI Application - Cursor, Claude Desktop, etc.)      │
│                                                             │
│  ┌───────────────────────────────────────────────────┐    │
│  │         MCP Client (Protocol Handler)             │    │
│  │  - Establishes connection                         │    │
│  │  - Sends tool discovery requests                  │    │
│  │  - Invokes tools                                  │    │
│  │  - Receives results                               │    │
│  └──────────────────┬────────────────────────────────┘    │
│                     │                                       │
└─────────────────────┼───────────────────────────────────────┘
                      │
                      │ MCP Protocol Messages
                      │ (JSON-RPC over Transport)
                      │
┌─────────────────────▼───────────────────────────────────────┐
│  ┌───────────────────────────────────────────────────┐    │
│  │         Transport Layer                           │    │
│  │  - stdio (Standard Input/Output)                  │    │
│  │  - SSE (Server-Sent Events over HTTP)            │    │
│  │  - WebSockets (future)                           │    │
│  └──────────────────┬────────────────────────────────┘    │
│                     │                                       │
│  ┌─────────────────▼─────────────────────────────────┐    │
│  │         MCP Server (Tool Provider)                │    │
│  │  - Registers tools                                │    │
│  │  - Validates inputs                               │    │
│  │  - Executes tool logic                            │    │
│  │  - Returns formatted results                      │    │
│  └───────────────────────────────────────────────────┘    │
│                                                             │
│                  External Resources                         │
│  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌─────────┐    │
│  │Database │  │  APIs    │  │  Files  │  │Services │    │
│  └─────────┘  └──────────┘  └─────────┘  └─────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Component 1: MCP Host (Your AI Application)

**What is it?**
The MCP Host is your AI application—the thing users interact with. Examples:
- Cursor IDE
- Claude Desktop
- Your custom chatbot
- Internal AI assistant

**Responsibilities:**
1. **Conversation Management:** Maintains chat history, user context
2. **LLM Orchestration:** Calls LLM (GPT-4, Claude, etc.) with messages
3. **Tool Decision:** LLM decides when to call tools (based on user query)
4. **MCP Client Management:** Instantiates and manages MCP clients
5. **Result Presentation:** Shows tool results to user

**Backend Engineering Analogy:** The MCP Host is like your **web application server** (Express, Django, FastAPI). It:
- Handles user requests (HTTP requests = user messages)
- Orchestrates business logic (LLM inference = business rules)
- Calls external services (MCP tools = microservices, databases)
- Returns responses to user

**Code Example (Conceptual):**
```typescript
class AIApplicationHost {
  private llm: LLMClient;
  private mcpClients: Map<string, MCPClient>;
  
  constructor() {
    this.llm = new OpenAIClient({ model: "gpt-4" });
    this.mcpClients = new Map();
    
    // Initialize MCP clients for various servers
    this.mcpClients.set("gmail", new MCPClient({
      serverCommand: "node",
      serverArgs: ["/path/to/gmail-mcp-server/index.js"]
    }));
    
    this.mcpClients.set("postgres", new MCPClient({
      serverUrl: "https://internal-tools.company.com/mcp/postgres"
    }));
  }
  
  async handleUserMessage(message: string): Promise<string> {
    // 1. Get all available tools from MCP servers
    const tools = await this.discoverAllTools();
    
    // 2. Send message to LLM with available tools
    const response = await this.llm.chat({
      messages: [{ role: "user", content: message }],
      tools: tools
    });
    
    // 3. If LLM wants to call a tool
    if (response.tool_calls) {
      const toolResults = await Promise.all(
        response.tool_calls.map(tc => this.executeTool(tc))
      );
      
      // 4. Send tool results back to LLM for final response
      const finalResponse = await this.llm.chat({
        messages: [
          { role: "user", content: message },
          { role: "assistant", tool_calls: response.tool_calls },
          ...toolResults.map(r => ({ role: "tool", content: r }))
        ]
      });
      
      return finalResponse.content;
    }
    
    return response.content;
  }
  
  private async discoverAllTools(): Promise<Tool[]> {
    const allTools = [];
    for (const [name, client] of this.mcpClients) {
      const tools = await client.listTools();
      allTools.push(...tools);
    }
    return allTools;
  }
  
  private async executeTool(toolCall: ToolCall): Promise<string> {
    const client = this.getClientForTool(toolCall.name);
    return await client.executeTool(toolCall.name, toolCall.arguments);
  }
}
```

### Component 2: MCP Client (Protocol Handler)

**What is it?**
The MCP Client is a **library/SDK** that handles the MCP protocol. It's embedded in your MCP Host.

**Responsibilities:**
1. **Connection Management:** Establishes and maintains connection to MCP server
2. **Message Serialization:** Converts function calls to JSON-RPC messages
3. **Protocol Handling:** Sends `tools/list`, `tools/call`, handles responses
4. **Error Handling:** Retries, timeouts, connection errors
5. **Transport Abstraction:** Works over stdio, SSE, WebSockets, etc.

**Backend Engineering Analogy:** The MCP Client is like an **HTTP client library** (axios, requests, fetch). You don't manually craft HTTP packets; the library does it.

**Key Insight:** You **don't write** MCP clients from scratch (unless you're building a new language SDK). You use existing SDKs:
- `@modelcontextprotocol/sdk` (TypeScript/JavaScript)
- `mcp` (Python)
- Community SDKs (Java, Rust, Go—growing)

**Code Example (Using MCP Client SDK):**
```typescript
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

class MCPClient {
  private client: Client;
  
  async connect(serverCommand: string, serverArgs: string[]) {
    // Transport handles stdio communication
    const transport = new StdioClientTransport({
      command: serverCommand,
      args: serverArgs
    });
    
    // Client handles protocol
    this.client = new Client({
      name: "my-ai-app",
      version: "1.0.0"
    }, {
      capabilities: {}
    });
    
    await this.client.connect(transport);
  }
  
  async listTools(): Promise<Tool[]> {
    // Client SDK handles JSON-RPC message construction
    const response = await this.client.request({
      method: "tools/list"
    }, ListToolsResultSchema);
    
    return response.tools;
  }
  
  async executeTool(name: string, args: any): Promise<string> {
    const response = await this.client.request({
      method: "tools/call",
      params: { name, arguments: args }
    }, CallToolResultSchema);
    
    // Extract text from response
    return response.content
      .filter(c => c.type === "text")
      .map(c => c.text)
      .join("\n");
  }
}
```

**What You Don't Have to Worry About:**
- JSON-RPC message formatting
- Request ID management
- Transport-specific details (stdio vs SSE)
- Protocol versioning
- Capability negotiation

### Component 3: MCP Server (Tool Provider)

**What is it?**
The MCP Server is a **standalone process or service** that provides tools to AI applications.

**Responsibilities:**
1. **Tool Registration:** Declares what tools it provides
2. **Input Validation:** Validates tool parameters (types, required fields)
3. **Tool Execution:** Runs the actual tool logic
4. **Result Formatting:** Returns results in MCP-compliant format
5. **Resource Management:** Manages connections to databases, APIs, files

**Backend Engineering Analogy:** The MCP Server is like a **REST API microservice**. It:
- Exposes endpoints (tools)
- Validates requests (input schemas)
- Executes business logic
- Returns responses
- Manages resources (DB connections, API clients)

**Key Insight:** MCP servers should be **thin wrappers**. The actual business logic (database queries, API calls) happens in external systems. The MCP server just provides the interface.

**Code Example (TypeScript MCP Server):**
```typescript
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import axios from "axios";

// Initialize server
const server = new Server({
  name: "weather-server",
  version: "1.0.0"
}, {
  capabilities: {
    tools: {}
  }
});

// Define tool schema
const GetWeatherSchema = z.object({
  city: z.string().describe("Name of the city")
});

// Register tool
server.setRequestHandler("tools/list", async () => {
  return {
    tools: [
      {
        name: "get_weather",
        description: "Get current weather for a city",
        inputSchema: {
          type: "object",
          properties: {
            city: {
              type: "string",
              description: "Name of the city"
            }
          },
          required: ["city"]
        }
      }
    ]
  };
});

// Handle tool execution
server.setRequestHandler("tools/call", async (request) => {
  const { name, arguments: args } = request.params;
  
  if (name === "get_weather") {
    // Validate input
    const { city } = GetWeatherSchema.parse(args);
    
    // Execute tool logic (call external API)
    const response = await axios.get(
      `https://api.openweathermap.org/data/2.5/weather`,
      {
        params: {
          q: city,
          appid: process.env.OPENWEATHER_API_KEY,
          units: "metric"
        }
      }
    );
    
    // Format result
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            city: city,
            temperature: response.data.main.temp,
            description: response.data.weather[0].description
          })
        }
      ]
    };
  }
  
  throw new Error(`Unknown tool: ${name}`);
});

// Start server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Weather MCP server running on stdio");
}

main().catch(console.error);
```

**Important Design Principles:**
1. **Stateless:** Each tool call is independent (no shared state between calls)
2. **Thin Wrapper:** Don't implement complex business logic in MCP server
3. **Error Handling:** Return clear error messages (don't crash the server)
4. **No Credentials:** Accept credentials from host, don't hardcode

### Component 4: Transport Layer

**What is it?**
The transport layer is **how** MCP messages physically move between client and server.

**Available Transports (as of Jan 2026):**
1. **stdio (Standard Input/Output):** Local process communication
2. **SSE (Server-Sent Events):** HTTP-based, server-to-client streaming

**Future Transports:**
3. **WebSockets:** Bidirectional, low-latency
4. **gRPC:** High-performance, strongly typed

#### Transport 1: stdio (Local Communication)

**How it works:**
```
┌─────────────────────┐
│   MCP Host Process  │
│                     │
│  ┌──────────────┐  │
│  │  MCP Client  │  │
│  └───────┬──────┘  │
│          │         │
│          │ spawn   │
│          ▼         │
│  ┌──────────────┐  │
│  │Child Process │  │ ← MCP Server (node index.js)
│  │(MCP Server)  │  │
│  │              │  │
│  │ stdin  ◄─────┼──┼─── JSON-RPC requests
│  │ stdout ──────┼──┼──► JSON-RPC responses
│  └──────────────┘  │
└─────────────────────┘
```

**Backend Engineering Analogy:** This is like **inter-process communication (IPC)** via pipes. Think of Unix pipes:
```bash
cat file.txt | grep "pattern" | wc -l
```
Each process reads from stdin, writes to stdout.

**When to use stdio:**
- ✓ MCP server runs on same machine as AI application
- ✓ Low latency critical (no network overhead)
- ✓ Simple deployment (just run a command)
- ✗ Cannot share servers across machines
- ✗ Harder to scale horizontally

**Configuration Example (Cursor):**
```json
{
  "mcpServers": {
    "gmail": {
      "command": "node",
      "args": ["/Users/you/.mcp-servers/gmail/index.js"]
    },
    "postgres": {
      "command": "python",
      "args": ["/Users/you/.mcp-servers/postgres/server.py"]
    }
  }
}
```

**How the connection works:**
1. MCP Host spawns child process: `node /path/to/server.js`
2. Child process starts, listens on stdin
3. MCP Client sends JSON-RPC message to child's stdin
4. MCP Server reads from stdin, processes, writes to stdout
5. MCP Client reads from child's stdout

**Key Insight:** stdio is **not** a network protocol. It's process-level communication. This is why you can't connect to a remote stdio MCP server.

#### Transport 2: SSE (Server-Sent Events)

**How it works:**
```
┌──────────────────────┐        HTTP Network        ┌──────────────────────┐
│   MCP Host Process   │                             │  MCP Server Process  │
│                      │                             │                      │
│  ┌───────────────┐  │    POST /messages          │  ┌───────────────┐  │
│  │  MCP Client   │──┼──────────────────────────►  │  │  HTTP Server  │  │
│  │               │  │                             │  │  (Express)    │  │
│  │               │  │                             │  └───────┬───────┘  │
│  │               │  │    GET /sse (persistent)   │          │          │
│  │               │◄─┼────────────────────────────┼──────────┘          │
│  │               │  │    ◄── SSE stream ◄──      │                      │
│  └───────────────┘  │                             │  ┌───────────────┐  │
│                      │                             │  │  MCP Server   │  │
│                      │                             │  │  Logic        │  │
│                      │                             │  └───────────────┘  │
└──────────────────────┘                             └──────────────────────┘
```

**Backend Engineering Analogy:** SSE is like **webhook + long-polling**:
- **POST /messages:** Client sends commands (like webhooks)
- **GET /sse:** Server pushes responses (like long-polling, but better)

**When to use SSE:**
- ✓ MCP server hosted remotely (different machine/container)
- ✓ Multiple AI apps connect to same server (horizontal scaling)
- ✓ Centralized tool management (one server, many clients)
- ✓ Firewall-friendly (just HTTP/HTTPS)
- ✗ Slightly higher latency than stdio (~10-50ms)

**Code Example (MCP Server with SSE):**
```typescript
import express from "express";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";

const app = express();
app.use(express.json());

const server = new Server({
  name: "remote-weather-server",
  version: "1.0.0"
}, {
  capabilities: { tools: {} }
});

// Register tools (same as before)
server.setRequestHandler("tools/list", async () => { ... });
server.setRequestHandler("tools/call", async (req) => { ... });

// SSE endpoint (persistent connection)
app.get("/sse", async (req, res) => {
  const transport = new SSEServerTransport("/messages", res);
  await server.connect(transport);
  console.log("Client connected via SSE");
});

// Message endpoint (client sends commands here)
app.post("/messages", async (req, res) => {
  // Transport handles routing to appropriate connection
  // (SSE connection is already established via /sse)
  res.status(200).send();
});

app.listen(3000, () => {
  console.log("MCP server listening on http://localhost:3000");
});
```

**Client Configuration:**
```json
{
  "mcpServers": {
    "weather": {
      "url": "https://mcp-server.company.com/weather"
    }
  }
}
```

**Trade-offs:**

| Aspect | stdio | SSE |
|--------|-------|-----|
| **Latency** | ~1-5ms | ~10-50ms |
| **Scalability** | One process per client | One server, many clients |
| **Deployment** | Simple (just a command) | Requires HTTP server |
| **Network** | Local only | Remote-capable |
| **Firewall** | No concerns | Must open HTTP port |
| **Debugging** | Harder (process logs) | Easier (HTTP inspection) |
| **Auth** | Not needed (local trust) | Required (API keys, OAuth) |

### The Complete Request Flow

Let's trace a full request through all components:

```mermaid
sequenceDiagram
    participant User
    participant Host as MCP Host<br/>(AI App)
    participant Client as MCP Client<br/>(SDK)
    participant Transport as Transport<br/>(stdio/SSE)
    participant Server as MCP Server<br/>(Tool Provider)
    participant External as External API<br/>(OpenWeather)

    Note over User,External: Step 1: Connection & Discovery
    Host->>Client: Initialize MCP client
    Client->>Transport: Establish connection
    Transport->>Server: Connect (spawn process or HTTP)
    Server-->>Transport: Connection ready
    Transport-->>Client: Connected
    
    Client->>Transport: Send: {"method": "tools/list"}
    Transport->>Server: Forward request
    Server-->>Transport: {"tools": [{"name": "get_weather", ...}]}
    Transport-->>Client: Tool list
    Client-->>Host: Available tools

    Note over User,External: Step 2: User Interaction
    User->>Host: "What's the weather in Mumbai?"
    Host->>Host: Call LLM with tools
    Host->>Host: LLM decides: call get_weather("Mumbai")

    Note over User,External: Step 3: Tool Execution
    Host->>Client: Execute tool: get_weather("Mumbai")
    Client->>Transport: {"method": "tools/call", "params": {...}}
    Transport->>Server: Forward request
    Server->>Server: Validate input
    Server->>External: GET /weather?q=Mumbai
    External-->>Server: {"temp": 28, "desc": "Cloudy"}
    Server-->>Transport: {"content": [{"type": "text", "text": "..."}]}
    Transport-->>Client: Tool result
    Client-->>Host: Result data

    Note over User,External: Step 4: Final Response
    Host->>Host: Send result to LLM
    Host->>Host: LLM formats answer
    Host-->>User: "The weather in Mumbai is 28°C and cloudy"
```

---

### Checkpoint Question 2: Transport Selection

**FAANG-Style Question:**

You're designing an AI platform for a financial services company with the following requirements:

**System Context:**
- 1000+ internal analysts use AI assistant daily
- Need access to 50+ data sources (databases, APIs, internal services)
- Strict security: Tools must run in isolated VPCs
- Compliance: All tool calls must be audited
- High availability: 99.9% uptime SLA

**Tool Requirements:**
- 20 tools are computationally expensive (run ML models, process large datasets)
- 30 tools are lightweight (database queries, API calls)
- Some tools access highly sensitive PII data
- Tools are maintained by different teams (10+ engineering teams)

**Question:**
1. Would you use **stdio** or **SSE** transport? Justify with specific architectural reasoning.
2. Design the deployment architecture (where do MCP servers run?)
3. How would you handle authentication and audit logging?
4. What are the latency and cost implications of your choice?

**Expected Answer:**

```mermaid
graph TB
    subgraph "User Network"
        A[Analyst Workstation<br/>MCP Host]
    end
    
    subgraph "DMZ / API Gateway"
        B[MCP Gateway<br/>Authentication & Routing]
        C[Audit Log Service]
    end
    
    subgraph "Internal VPC - Lightweight Tools"
        D1[MCP Server: Postgres]
        D2[MCP Server: API Calls]
        D3[MCP Server: Redis]
    end
    
    subgraph "Internal VPC - Heavy Tools"
        E1[MCP Server: ML Models<br/>GPU Instance]
        E2[MCP Server: Data Processing<br/>High-Memory]
    end
    
    subgraph "Sensitive Data VPC"
        F1[MCP Server: PII Database<br/>Extra Security]
    end
    
    A -->|SSE over HTTPS| B
    B -->|Log all requests| C
    B -->|Route by tool type| D1
    B --> D2
    B --> D3
    B -->|Route heavy tools| E1
    B --> E2
    B -->|Extra auth check| F1
```

**Answer:**

**1. Transport Choice: SSE (Strongly Recommended)**

**Why NOT stdio:**
- ❌ **Security Isolation:** stdio requires MCP server to run on same machine as AI app (analyst workstation). This means:
  - Sensitive credentials on analyst laptops
  - No network isolation between tools
  - Can't enforce VPC boundaries
- ❌ **Resource Management:** Heavy ML tools would consume analyst workstation resources
- ❌ **Centralized Updates:** Updating 50 tools means deploying to 1000+ workstations
- ❌ **Audit Trail:** Harder to centralize audit logs from 1000 processes

**Why YES to SSE:**
- ✅ **Security Isolation:** MCP servers run in controlled VPCs
- ✅ **Centralized Management:** One deployment updates all users
- ✅ **Resource Optimization:** Heavy tools run on dedicated infra
- ✅ **Audit Trail:** All requests flow through central gateway
- ✅ **Compliance:** Can enforce data residency, access controls

**2. Deployment Architecture:**

```typescript
// MCP Gateway (Central Orchestrator)
class MCPGateway {
  private servers: Map<string, MCPServerConfig>;
  private authService: AuthService;
  private auditLogger: AuditLogger;
  
  constructor() {
    this.servers = new Map([
      ["postgres", { 
        url: "https://internal-tools.company.com/mcp/postgres",
        vpc: "internal",
        sensitivity: "medium"
      }],
      ["pii_database", {
        url: "https://sensitive-tools.company.com/mcp/pii",
        vpc: "sensitive",
        sensitivity: "high",
        requiredRole: "data_analyst_senior"
      }],
      ["ml_model", {
        url: "https://compute-tools.company.com/mcp/ml",
        vpc: "compute",
        sensitivity: "medium",
        instanceType: "gpu"
      }]
    ]);
  }
  
  async executeToolWithGovernance(
    userId: string,
    toolName: string,
    params: any
  ): Promise<any> {
    // 1. Authentication
    const user = await this.authService.validateUser(userId);
    if (!user) {
      throw new UnauthorizedError("Invalid user");
    }
    
    // 2. Authorization (tool-level)
    const serverConfig = this.servers.get(this.getServerForTool(toolName));
    if (serverConfig.requiredRole && 
        !user.roles.includes(serverConfig.requiredRole)) {
      await this.auditLogger.logUnauthorizedAccess({
        userId,
        toolName,
        reason: "Missing required role"
      });
      throw new ForbiddenError("Insufficient permissions");
    }
    
    // 3. Rate Limiting (per user, per tool category)
    const rateLimitKey = `${userId}:${serverConfig.sensitivity}`;
    if (await this.rateLimiter.isLimited(rateLimitKey)) {
      throw new RateLimitError("Too many requests");
    }
    
    // 4. Audit Logging (before execution)
    const requestId = generateRequestId();
    await this.auditLogger.logRequest({
      requestId,
      userId,
      toolName,
      params: this.sanitizeForAudit(params),
      timestamp: Date.now(),
      serverVpc: serverConfig.vpc
    });
    
    // 5. Execute tool (with timeout based on sensitivity)
    const timeout = serverConfig.sensitivity === "high" ? 5000 : 30000;
    try {
      const result = await this.executeWithTimeout(
        serverConfig.url,
        toolName,
        params,
        timeout
      );
      
      // 6. Audit logging (after success)
      await this.auditLogger.logSuccess({
        requestId,
        resultSize: JSON.stringify(result).length,
        duration: Date.now() - start
      });
      
      return result;
      
    } catch (error) {
      // 7. Audit logging (after failure)
      await this.auditLogger.logFailure({
        requestId,
        error: error.message,
        duration: Date.now() - start
      });
      throw error;
    }
  }
  
  private sanitizeForAudit(params: any): any {
    // Remove PII from audit logs
    const sanitized = { ...params };
    const piiFields = ["ssn", "credit_card", "password"];
    for (const field of piiFields) {
      if (sanitized[field]) {
        sanitized[field] = "***REDACTED***";
      }
    }
    return sanitized;
  }
}
```

**3. Authentication & Audit Strategy:**

**Authentication (Multi-Layer):**
```typescript
// Layer 1: User to Gateway
// - OAuth 2.0 with corporate IdP (Okta, Azure AD)
// - JWT tokens with 1-hour expiry
// - MFA required for sensitive tools

// Layer 2: Gateway to MCP Servers
// - Service-to-service auth (mTLS)
// - API keys rotated every 30 days
// - Network isolation (VPC peering)

// Layer 3: MCP Server to Data Sources
// - Database credentials from secrets manager (Vault)
// - IAM roles for cloud services (AWS/GCP)
// - Just-in-time credential provisioning
```

**Audit Logging (Comprehensive):**
```typescript
interface AuditLogEntry {
  requestId: string;
  timestamp: number;
  userId: string;
  userEmail: string;
  toolName: string;
  serverVpc: string;
  params: any;  // Sanitized
  resultSize: number;
  duration: number;
  success: boolean;
  errorMessage?: string;
  ipAddress: string;
  userAgent: string;
}

// Store in:
// - Primary: Time-series database (InfluxDB, TimescaleDB)
// - Compliance: Write-once storage (S3 Glacier, Azure Archive)
// - Real-time: Streaming to SIEM (Splunk, Datadog)
```

**4. Latency & Cost Analysis:**

**Latency Breakdown:**
```
Total Latency = Auth + Network + Execution + Logging

stdio (hypothetical local):
- Auth: 0ms (local trust)
- Network: 0ms
- Execution: 100ms (tool logic)
- Logging: 0ms
= 100ms total

SSE (actual design):
- Auth: 10ms (JWT validation + cache)
- Network: 20ms (internal network, same region)
- Execution: 100ms (tool logic)
- Logging: 5ms (async write to audit DB)
= 135ms total

Additional latency: 35ms (35% overhead)
```

**Is 35ms acceptable?**
- ✅ **Yes** for analyst workflows (humans don't notice <200ms)
- ✅ **Yes** compared to LLM inference (1000-3000ms)
- ✅ **Yes** for compliance benefits (mandatory audit trail)

**Cost Analysis:**

**stdio approach (if it were possible):**
- ❌ 1000 workstations × 50 tools = 50,000 process instances
- ❌ No resource sharing (GPU, memory wasted)
- ❌ High data egress costs (each workstation pulls data)

**SSE approach (actual):**
```
Infrastructure:
- MCP Gateway: 4 instances (HA) × $100/month = $400
- Lightweight MCP Servers: 10 instances × $50 = $500
- Heavy MCP Servers (GPU): 3 instances × $500 = $1,500
- Audit Log Database: $300/month
- Total: $2,700/month

Per-User Cost: $2,700 / 1,000 users = $2.70/user/month
```

**Cost Benefit:**
- Centralized resources save ~60% on compute
- Audit trail prevents compliance fines (potentially millions)
- Faster updates reduce engineering time (50+ tools × 1000 workstations = 50,000 deployments avoided)

**Key Trade-Off Decision:**
> Accept 35ms latency overhead for 99% reduction in security risk, 60% cost savings, and 100% audit coverage.

---

## Step 5: Transport Mechanisms Deep Dive

### Why Transport Matters

In distributed systems, the transport layer is often an afterthought. But in MCP, **transport selection fundamentally shapes your architecture**.

Think about HTTP. Would microservices work without HTTP? Would REST APIs work over Unix domain sockets? The transport dictates:
- **Where** components can run (same process, same machine, different datacenter)
- **How** you scale (vertical, horizontal)
- **What** security model you need (local trust, network auth)

### stdio Transport: The Local IPC Pattern

#### The Technical Mechanism

When you configure an MCP server with stdio transport:

```json
{
  "mcpServers": {
    "weather": {
      "command": "node",
      "args": ["/path/to/weather-server/index.js"]
    }
  }
}
```

**What actually happens:**

1. **Process Spawning:**
```typescript
// MCP Host spawns child process
const child = spawn("node", ["/path/to/weather-server/index.js"], {
  stdio: ['pipe', 'pipe', 'inherit']
  //       stdin   stdout  stderr (inherit = parent's stderr)
});
```

2. **Pipe Setup:**
```
Parent Process (MCP Host)     Child Process (MCP Server)
┌─────────────────────┐       ┌─────────────────────┐
│                     │       │                     │
│  child.stdin.write()│──────►│  process.stdin      │
│                     │       │       ↓             │
│                     │       │  Read JSON-RPC      │
│                     │       │  Parse & Execute    │
│                     │       │       ↓             │
│  child.stdout.on()  │◄──────│  process.stdout     │
│                     │       │                     │
└─────────────────────┘       └─────────────────────┘
```

3. **Message Flow:**
```typescript
// MCP Host (Parent) sends request
const request = {
  jsonrpc: "2.0",
  method: "tools/list",
  id: 1
};
child.stdin.write(JSON.stringify(request) + "\n");

// MCP Server (Child) receives
process.stdin.on('data', (chunk) => {
  const request = JSON.parse(chunk.toString());
  // Process request...
});

// MCP Server (Child) sends response
const response = {
  jsonrpc: "2.0",
  result: { tools: [...] },
  id: 1
};
process.stdout.write(JSON.stringify(response) + "\n");

// MCP Host (Parent) receives
child.stdout.on('data', (chunk) => {
  const response = JSON.parse(chunk.toString());
  // Handle response...
});
```

**Backend Engineering Analogy:** This is exactly how Unix pipes work:
```bash
echo '{"city": "Mumbai"}' | node weather-server.js
```

The parent writes to child's stdin, child writes to stdout. It's **synchronous IPC**.

#### Performance Characteristics

**Latency Breakdown:**
```
stdio message passing:
- JSON serialization: ~0.1ms (small payloads)
- Pipe write (syscall): ~0.01ms
- Context switch: ~0.005ms
- Pipe read (syscall): ~0.01ms
- JSON deserialization: ~0.1ms
= ~0.225ms total

Compare to HTTP localhost:
- TCP handshake: ~0.1ms
- HTTP parsing: ~0.2ms
- Kernel routing: ~0.05ms
= ~0.35ms minimum
```

**stdio is 50% faster than HTTP on localhost.**

**Memory Characteristics:**
```typescript
// Each MCP server is a separate process
const weatherServer = spawn("node", ["weather.js"]);  // ~50MB Node.js process
const gmailServer = spawn("node", ["gmail.js"]);      // ~50MB Node.js process
const postgresServer = spawn("python", ["db.js"]);    // ~40MB Python process

// Total: 140MB for 3 servers
// Each has its own heap, GC, runtime
```

**Backend Engineering Analogy:** This is like running separate Docker containers vs threads.
- **Isolation:** Each server crashes independently
- **Overhead:** Process-level isolation costs memory
- **Security:** OS enforces process boundaries

#### When stdio Shines

**Scenario 1: Desktop Applications**

Cursor IDE uses stdio for all MCP servers:

```typescript
// Cursor's MCP configuration
{
  "mcpServers": {
    "filesystem": {
      "command": "node",
      "args": ["~/.cursor/mcp-servers/filesystem/index.js"]
    },
    "git": {
      "command": "node",
      "args": ["~/.cursor/mcp-servers/git/index.js"]
    },
    "terminal": {
      "command": "python",
      "args": ["~/.cursor/mcp-servers/terminal/server.py"]
    }
  }
}
```

**Why this works:**
- ✅ User's machine has all dependencies (Node.js, Python)
- ✅ Tools access local filesystem (no network needed)
- ✅ Ultra-low latency (local commands like `git status`)
- ✅ Simple installation (`npm install -g @cursor/mcp-filesystem`)
- ✅ No firewall/port configuration

**Scenario 2: CLI Tools**

```bash
# AI assistant as a CLI tool
ai "Send email to john@example.com"

# Uses stdio MCP servers for:
# - Email (local SMTP client)
# - Calendar (local iCal files)
# - Notes (local markdown files)
```

**Why this works:**
- ✅ All data is local
- ✅ No need for server infrastructure
- ✅ Works offline
- ✅ Fast startup (no network handshake)

#### When stdio Fails

**Scenario 1: Multi-User SaaS**

You're building an AI assistant for 10,000 users:

```typescript
// ❌ WRONG: Each user gets their own MCP server processes
app.post('/chat', async (req, res) => {
  const userId = req.user.id;
  
  // Spawn process per user?
  const weatherServer = spawn("node", ["weather.js"]);
  const gmailServer = spawn("node", ["gmail.js"]);
  
  // 10,000 users × 2 servers = 20,000 processes
  // ~50MB × 20,000 = 1TB RAM (!!)
});
```

**Why this fails:**
- ❌ Cannot share servers across users (no network)
- ❌ 1TB RAM for 10,000 users
- ❌ Cannot scale horizontally (processes tied to single machine)
- ❌ Deployment nightmare (update 20,000 processes)

**Scenario 2: Containerized Environments**

```yaml
# Docker Compose with stdio (problematic)
services:
  ai-app:
    image: my-ai-app
    # Where do MCP servers run?
    # - In same container? (breaks isolation)
    # - In separate containers? (stdio can't cross containers!)
```

**Why this fails:**
- ❌ stdio cannot cross container boundaries
- ❌ Kubernetes can't orchestrate process spawning
- ❌ No service discovery (hardcoded paths)
- ❌ Cannot health-check child processes easily

### SSE Transport: The Network-Ready Pattern

#### The Technical Mechanism

SSE (Server-Sent Events) uses HTTP for bidirectional communication:

```typescript
// MCP Server (Express.js)
import express from "express";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";

const app = express();
app.use(express.json());

// Persistent connection endpoint
app.get("/sse", async (req, res) => {
  // Set SSE headers
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  
  // Create transport
  const transport = new SSEServerTransport("/messages", res);
  await server.connect(transport);
  
  // Connection stays open until client disconnects
  req.on('close', () => {
    console.log('Client disconnected');
  });
});

// Message ingress endpoint
app.post("/messages", async (req, res) => {
  // MCP client sends commands here
  // Transport routes to appropriate SSE connection
  res.status(202).json({ status: "accepted" });
});

app.listen(3000);
```

**How SSE works:**

```
Step 1: Client establishes SSE connection
┌─────────────┐                    ┌─────────────┐
│ MCP Client  │───GET /sse────────►│ MCP Server  │
│             │                    │             │
│             │◄───HTTP 200────────│             │
│             │    Content-Type:   │             │
│             │    text/event-     │             │
│             │    stream          │             │
└─────────────┘                    └─────────────┘
       │                                  │
       │         Connection stays open    │
       │◄─────────────────────────────────┤
       │                                  │
       
Step 2: Client sends command (separate HTTP request)
┌─────────────┐                    ┌─────────────┐
│ MCP Client  │─POST /messages────►│ MCP Server  │
│             │ {"method":         │             │
│             │  "tools/list"}     │             │
│             │                    │             │
│             │◄───202 Accepted────│             │
└─────────────┘                    └─────────────┘

Step 3: Server pushes response via SSE
┌─────────────┐                    ┌─────────────┐
│ MCP Client  │                    │ MCP Server  │
│             │◄─SSE event─────────│             │
│             │  data: {"jsonrpc": │             │
│             │   "2.0", "result": │             │
│             │   {...}}           │             │
└─────────────┘                    └─────────────┘
```

**The SSE Message Format:**
```http
GET /sse HTTP/1.1
Host: mcp-server.company.com

HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive

data: {"jsonrpc":"2.0","result":{"tools":[...]},"id":1}

data: {"jsonrpc":"2.0","result":{"content":[...]},"id":2}

data: {"jsonrpc":"2.0","result":{"content":[...]},"id":3}

```

**Backend Engineering Analogy:** SSE is like **WebSockets Lite**:
- WebSockets: Full duplex (bidirectional over single connection)
- SSE: Simplex (server → client) + HTTP POST (client → server)

#### Why Two Endpoints?

**Question:** Why `GET /sse` AND `POST /messages`? Why not just WebSockets?

**Answer:** Firewall compatibility.

```
Traditional Network Security:
┌─────────────────────────────────────────┐
│  Corporate Firewall                     │
│                                         │
│  ✅ Allow: HTTP GET/POST (port 80/443) │
│  ❌ Block: WebSocket upgrades          │
│  ❌ Block: Custom protocols             │
└─────────────────────────────────────────┘
```

SSE uses standard HTTP:
- `GET /sse` → Regular HTTP GET (no protocol upgrade)
- `POST /messages` → Regular HTTP POST
- Both work through standard HTTP proxies

**This is why SSE was chosen over WebSockets for MCP v1.0.**

#### Performance Characteristics

**Latency Breakdown:**
```
SSE message passing (same datacenter):
- HTTP POST to /messages: ~5-10ms
- Server processes request: ~0.1ms
- SSE event push: ~5-10ms
= ~10-20ms total

Cross-region:
- HTTP POST: ~50-100ms (network)
- Processing: ~0.1ms
- SSE push: ~50-100ms
= ~100-200ms total
```

**Connection Overhead:**
```typescript
// Each SSE connection holds:
// - HTTP connection: ~4KB (TCP buffers)
// - Event stream buffer: ~16KB
// - Application state: varies
= ~20KB per connection

// 10,000 concurrent users:
10,000 × 20KB = ~200MB
// Much better than stdio (1TB)
```

#### When SSE Shines

**Scenario 1: Multi-Tenant SaaS**

```typescript
// Single MCP server, many clients
const mcpServer = new MCPServer({
  name: "shared-weather-server",
  version: "1.0.0"
});

app.get("/sse", async (req, res) => {
  const userId = req.headers['x-user-id'];
  
  // Validate user
  const user = await authService.validate(userId);
  if (!user) return res.status(401).send("Unauthorized");
  
  // Create isolated transport with user context
  const transport = new SSEServerTransport("/messages", res);
  const userScopedServer = mcpServer.withContext({ userId });
  
  await userScopedServer.connect(transport);
  
  // One process serves 10,000 users
  console.log(`Active connections: ${connectionCount++}`);
});
```

**Why this works:**
- ✅ One server process serves all users
- ✅ ~200MB for 10,000 users (vs 1TB with stdio)
- ✅ Horizontal scaling (load balancer → multiple server instances)
- ✅ Centralized updates (deploy once, all users benefit)

**Scenario 2: Kubernetes/Cloud Native**

```yaml
# docker-compose.yml
services:
  ai-app:
    image: my-ai-app
    environment:
      MCP_WEATHER_URL: http://weather-mcp:3000
      MCP_GMAIL_URL: http://gmail-mcp:3001
  
  weather-mcp:
    image: my-weather-mcp-server
    ports:
      - "3000:3000"
  
  gmail-mcp:
    image: my-gmail-mcp-server
    ports:
      - "3001:3001"
```

```yaml
# kubernetes.yml
apiVersion: v1
kind: Service
metadata:
  name: weather-mcp
spec:
  selector:
    app: weather-mcp
  ports:
    - protocol: TCP
      port: 80
      targetPort: 3000
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: weather-mcp
spec:
  replicas: 3  # Horizontal scaling!
  template:
    spec:
      containers:
      - name: weather-mcp
        image: my-weather-mcp-server:v1.2.3
```

**Why this works:**
- ✅ Service discovery (Kubernetes DNS)
- ✅ Health checks (HTTP endpoints)
- ✅ Rolling updates (zero downtime)
- ✅ Autoscaling (HPA based on CPU/memory)
- ✅ Load balancing (built-in)

#### Real-World Hybrid Architecture

Most production systems use **both** transports:

```typescript
class SmartMCPHost {
  async initializeMCPClients() {
    // Local tools: stdio (fast, isolated)
    this.mcpClients.set("filesystem", new MCPClient({
      transport: "stdio",
      command: "node",
      args: ["~/.mcp/filesystem/index.js"]
    }));
    
    this.mcpClients.set("git", new MCPClient({
      transport: "stdio",
      command: "node",
      args: ["~/.mcp/git/index.js"]
    }));
    
    // Shared tools: SSE (centralized, scalable)
    this.mcpClients.set("weather", new MCPClient({
      transport: "sse",
      url: "https://mcp.company.com/weather"
    }));
    
    this.mcpClients.set("database", new MCPClient({
      transport: "sse",
      url: "https://mcp.company.com/database",
      headers: {
        Authorization: `Bearer ${await this.getAuthToken()}`
      }
    }));
  }
}
```

**Decision Matrix:**

| Tool Type | Data Location | Multi-User? | Transport Choice |
|-----------|---------------|-------------|------------------|
| Filesystem access | Local disk | No | stdio |
| Git operations | Local repo | No | stdio |
| Terminal commands | Local shell | No | stdio |
| Weather API | External | Yes | SSE |
| Company database | Internal | Yes | SSE |
| Email (user's) | External | No | stdio (local OAuth) |
| Email (shared) | External | Yes | SSE (centralized) |

---

## Step 6: Tool Lifecycle - Discovery, Invocation, Execution

### The Three-Phase Protocol

Every MCP interaction follows a strict lifecycle:

```
Phase 1: Connection & Discovery
│
├─ MCP Client connects to MCP Server
├─ Client sends: tools/list
├─ Server responds: Array of tool definitions
└─ Client caches tool schemas
│
Phase 2: Tool Invocation (repeated)
│
├─ User sends message to AI
├─ LLM decides to call tool
├─ MCP Host instructs MCP Client
├─ Client sends: tools/call with parameters
├─ Server validates & executes
├─ Server responds: Result data
└─ MCP Host sends result to LLM
│
Phase 3: Disconnection (cleanup)
│
├─ Client closes connection
├─ Server releases resources
└─ Cleanup complete
```

### Phase 1: Tool Discovery

#### The Discovery Protocol

When an MCP client connects, the **first** thing it does is discover available tools:

```typescript
// MCP Client (automatic discovery)
class MCPClient {
  async connect() {
    // Establish connection
    await this.transport.connect();
    
    // Immediately discover tools
    const response = await this.request({
      jsonrpc: "2.0",
      method: "tools/list",
      params: {},  // No parameters needed
      id: this.generateId()
    });
    
    // Cache tool definitions
    this.toolCache = new Map();
    for (const tool of response.result.tools) {
      this.toolCache.set(tool.name, tool);
    }
    
    console.log(`Discovered ${response.result.tools.length} tools`);
  }
}
```

**The `tools/list` Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/list",
  "id": 1
}
```

**The `tools/list` Response:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "tools": [
      {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "inputSchema": {
          "type": "object",
          "properties": {
            "city": {
              "type": "string",
              "description": "Name of the city"
            },
            "units": {
              "type": "string",
              "enum": ["metric", "imperial"],
              "description": "Temperature units",
              "default": "metric"
            }
          },
          "required": ["city"]
        }
      },
      {
        "name": "get_forecast",
        "description": "Get 5-day weather forecast",
        "inputSchema": {
          "type": "object",
          "properties": {
            "city": { "type": "string" },
            "days": {
              "type": "number",
              "minimum": 1,
              "maximum": 5,
              "default": 5
            }
          },
          "required": ["city"]
        }
      }
    ]
  },
  "id": 1
}
```

#### The Schema Format (JSON Schema)

MCP uses **JSON Schema** for input validation:

```typescript
// Example: Complex tool with nested objects
{
  "name": "create_calendar_event",
  "description": "Create a new calendar event",
  "inputSchema": {
    "type": "object",
    "properties": {
      "title": {
        "type": "string",
        "minLength": 1,
        "maxLength": 200
      },
      "start": {
        "type": "string",
        "format": "date-time",
        "description": "ISO 8601 format"
      },
      "end": {
        "type": "string",
        "format": "date-time"
      },
      "attendees": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "email": { "type": "string", "format": "email" },
            "optional": { "type": "boolean", "default": false }
          },
          "required": ["email"]
        },
        "maxItems": 100
      },
      "recurrence": {
        "type": "object",
        "properties": {
          "frequency": {
            "type": "string",
            "enum": ["daily", "weekly", "monthly"]
          },
          "interval": {
            "type": "number",
            "minimum": 1,
            "maximum": 30
          },
          "until": {
            "type": "string",
            "format": "date"
          }
        },
        "required": ["frequency"]
      }
    },
    "required": ["title", "start", "end"]
  }
}
```

**Backend Engineering Analogy:** This is exactly like **OpenAPI/Swagger schemas**:
- Defines parameter types
- Specifies required vs optional
- Provides validation rules
- Includes human-readable descriptions

#### How LLMs Use Tool Schemas

**The Magic:** LLMs can read JSON Schema and understand what parameters to provide.

```typescript
// Step 1: MCP Host sends tools to LLM
const response = await openai.chat.completions.create({
  model: "gpt-4",
  messages: [
    { role: "user", content: "What's the weather in Paris?" }
  ],
  tools: [
    {
      type: "function",
      function: {
        name: "get_weather",
        description: "Get current weather for a city",
        parameters: {  // This is the JSON Schema from MCP!
          type: "object",
          properties: {
            city: { type: "string" }
          },
          required: ["city"]
        }
      }
    }
  ]
});

// Step 2: LLM responds with tool call
// {
//   role: "assistant",
//   tool_calls: [{
//     id: "call_abc123",
//     function: {
//       name: "get_weather",
//       arguments: '{"city": "Paris"}'  // LLM extracted "Paris"!
//     }
//   }]
// }
```

**Key Insight:** The LLM:
1. Reads tool descriptions
2. Understands parameter schemas
3. Extracts relevant data from user message
4. Formats it correctly (JSON)
5. Returns structured tool call

This is why **good tool descriptions** matter:

```typescript
// ❌ BAD: Vague description
{
  "name": "query",
  "description": "Query something",
  "inputSchema": {
    "type": "object",
    "properties": {
      "q": { "type": "string" }
    }
  }
}

// ✅ GOOD: Clear, specific description
{
  "name": "query_customer_database",
  "description": "Search customer records by name, email, or customer ID. Returns up to 50 matching records. Requires exact match for email, partial match for name.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "search_term": {
        "type": "string",
        "description": "Customer name, email, or ID to search for"
      },
      "search_type": {
        "type": "string",
        "enum": ["name", "email", "id"],
        "description": "Type of search to perform"
      },
      "limit": {
        "type": "number",
        "minimum": 1,
        "maximum": 50,
        "default": 10,
        "description": "Maximum number of results to return"
      }
    },
    "required": ["search_term", "search_type"]
  }
}
```

The LLM can now:
- Understand this is a customer search
- Know it can search by name, email, or ID
- Understand there's a 50-record limit
- Ask user for clarification if needed

### Phase 2: Tool Invocation

#### The Invocation Protocol

When the LLM decides to call a tool:

```typescript
// MCP Client invokes tool
const result = await mcpClient.executeTool("get_weather", {
  city: "Paris",
  units: "metric"
});
```

**The `tools/call` Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "get_weather",
    "arguments": {
      "city": "Paris",
      "units": "metric"
    }
  },
  "id": 2
}
```

**The `tools/call` Response:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"temperature\": 18, \"description\": \"Partly cloudy\", \"humidity\": 65}"
      }
    ]
  },
  "id": 2
}
```

#### Server-Side Validation & Execution

```typescript
// MCP Server (validation + execution)
server.setRequestHandler("tools/call", async (request) => {
  const { name, arguments: args } = request.params;
  
  // Step 1: Find tool definition
  const tool = this.tools.get(name);
  if (!tool) {
    throw new Error(`Unknown tool: ${name}`);
  }
  
  // Step 2: Validate input against JSON Schema
  const validator = new JSONSchemaValidator(tool.inputSchema);
  const validationResult = validator.validate(args);
  
  if (!validationResult.valid) {
    throw new Error(
      `Invalid parameters: ${validationResult.errors.join(', ')}`
    );
  }
  
  // Step 3: Execute tool logic
  try {
    const result = await tool.execute(args);
    
    // Step 4: Format result
    return {
      content: [
        {
          type: "text",
          text: typeof result === "string" 
            ? result 
            : JSON.stringify(result)
        }
      ]
    };
  } catch (error) {
    // Step 5: Handle errors gracefully
    return {
      content: [
        {
          type: "text",
          text: `Error: ${error.message}`
        }
      ],
      isError: true
    };
  }
});
```

**Key Design Principle:** MCP servers should **never crash** on bad input. Always return a structured error.

#### Result Formats

MCP supports multiple content types:

```typescript
// 1. Text result (most common)
{
  content: [
    { type: "text", text: "The weather is sunny" }
  ]
}

// 2. Image result (base64 encoded)
{
  content: [
    {
      type: "image",
      data: "iVBORw0KGgoAAAANSUhEUgA...",
      mimeType: "image/png"
    }
  ]
}

// 3. Multiple content blocks
{
  content: [
    { type: "text", text: "Here's the chart:" },
    {
      type: "image",
      data: "base64_data_here",
      mimeType: "image/png"
    },
    { type: "text", text: "As you can see, sales are up 20%" }
  ]
}

// 4. Error result
{
  content: [
    {
      type: "text",
      text: "Error: API rate limit exceeded. Please try again in 60 seconds."
    }
  ],
  isError: true
}
```

**Backend Engineering Analogy:** This is like HTTP response bodies:
- `text/plain` → text content
- `image/png` → image content
- `multipart/mixed` → multiple content blocks

### Phase 3: The Complete Flow (Real-World Example)

Let's trace a complete interaction:

```mermaid
sequenceDiagram
    participant User
    participant AI as AI Application<br/>(MCP Host)
    participant Client as MCP Client<br/>(SDK)
    participant Server as MCP Server<br/>(Weather)
    participant API as OpenWeather<br/>API

    Note over User,API: Phase 1: Connection & Discovery
    AI->>Client: Initialize("weather-server")
    Client->>Server: Connect (stdio/SSE)
    Server-->>Client: Connected
    Client->>Server: {"method": "tools/list"}
    Server-->>Client: {"tools": [{"name": "get_weather", ...}]}
    Client-->>AI: Tools cached

    Note over User,API: Phase 2: User Interaction
    User->>AI: "What's the weather in Tokyo and Paris?"
    AI->>AI: Call LLM with available tools
    
    Note over AI: LLM decides to call tool TWICE<br/>(parallel execution)
    
    AI->>Client: executeTool("get_weather", {"city": "Tokyo"})
    AI->>Client: executeTool("get_weather", {"city": "Paris"})
    
    par Tokyo Request
        Client->>Server: {"method": "tools/call", "params": {"name": "get_weather", "arguments": {"city": "Tokyo"}}}
        Server->>Server: Validate input
        Server->>API: GET /weather?q=Tokyo
        API-->>Server: {"temp": 22, "desc": "Clear"}
        Server-->>Client: {"content": [{"type": "text", "text": "..."}]}
        Client-->>AI: Result: Tokyo is 22°C, Clear
    and Paris Request
        Client->>Server: {"method": "tools/call", "params": {"name": "get_weather", "arguments": {"city": "Paris"}}}
        Server->>Server: Validate input
        Server->>API: GET /weather?q=Paris
        API-->>Server: {"temp": 18, "desc": "Cloudy"}
        Server-->>Client: {"content": [{"type": "text", "text": "..."}]}
        Client-->>AI: Result: Paris is 18°C, Cloudy
    end
    
    AI->>AI: Send both results to LLM
    AI->>AI: LLM formats natural response
    AI-->>User: "Tokyo is 22°C and clear, while Paris is 18°C and cloudy."
```

---

### Checkpoint Question 3: Tool Design

**FAANG-Style Question:**

You're the principal engineer at a healthcare company building an AI assistant for doctors. The assistant needs to:

1. **Query patient records** (PostgreSQL database with 10M+ records)
2. **Search medical literature** (PubMed API, millions of papers)
3. **Generate treatment plans** (calls internal ML model, 5-10 seconds per request)
4. **Schedule appointments** (integrates with calendar system)

**Constraints:**
- HIPAA compliance (all PHI must be encrypted, logged)
- Sub-second latency for common queries (patient lookup)
- High availability (99.99% uptime)
- Cost optimization (minimize API calls, database queries)

**Question:**
1. Design the MCP server architecture (how many servers? what tools?)
2. How would you handle caching to reduce database load?
3. How would you handle the slow ML model (5-10s latency)?
4. Design the tool schemas (focus on parameter validation)

**Expected Answer:**

```mermaid
graph TB
    subgraph "MCP Host (AI Assistant)"
        A[AI Application]
        B[Cache Layer<br/>Redis]
    end
    
    subgraph "MCP Servers"
        C[Patient MCP Server<br/>Fast Queries]
        D[Literature MCP Server<br/>External API]
        E[Treatment MCP Server<br/>ML Model]
        F[Scheduling MCP Server<br/>Calendar API]
    end
    
    subgraph "Data Layer"
        G[PostgreSQL<br/>Patient Records]
        H[PubMed API]
        I[ML Model Service<br/>GPU]
        J[Calendar Service]
    end
    
    A -->|Check cache first| B
    A -->|SSE| C
    A -->|SSE| D
    A -->|SSE + Async| E
    A -->|SSE| F
    
    C --> G
    D --> H
    E --> I
    F --> J
    
    B -.->|Cache hit| A
```

**Answer:**

**1. MCP Server Architecture:**

```typescript
// Server 1: Patient MCP Server (Hot Path - Optimized)
class PatientMCPServer {
  tools = [
    {
      name: "search_patient_by_id",
      description: "Fast lookup by patient ID. Returns full patient record including demographics, medical history, and current medications. Average latency: 50ms.",
      inputSchema: {
        type: "object",
        properties: {
          patient_id: {
            type: "string",
            pattern: "^P[0-9]{8}$",  // Format: P12345678
            description: "8-digit patient ID with 'P' prefix"
          },
          include_history: {
            type: "boolean",
            default: false,
            description: "Include full medical history (slower)"
          }
        },
        required: ["patient_id"]
      }
    },
    {
      name: "search_patients_by_name",
      description: "Search patients by name. Returns up to 20 matches. Slower than ID search. Use for fuzzy matching.",
      inputSchema: {
        type: "object",
        properties: {
          first_name: { type: "string", minLength: 2 },
          last_name: { type: "string", minLength: 2 },
          date_of_birth: {
            type: "string",
            format: "date",
            description: "ISO 8601 date (YYYY-MM-DD) to narrow search"
          }
        },
        required: ["last_name"]  // First name optional for search
      }
    }
  ];
  
  async execute(toolName, args) {
    // Tool execution with caching
    const cacheKey = `${toolName}:${JSON.stringify(args)}`;
    
    // Check cache (Redis)
    const cached = await this.cache.get(cacheKey);
    if (cached) {
      return { ...cached, _cached: true };
    }
    
    // Execute query
    const result = await this.executeQuery(toolName, args);
    
    // Cache for 5 minutes (balance freshness vs performance)
    await this.cache.setEx(cacheKey, 300, result);
    
    return result;
  }
}

// Server 2: Literature MCP Server (External API)
class LiteratureMCPServer {
  tools = [
    {
      name: "search_medical_literature",
      description: "Search PubMed for relevant medical papers. Returns abstracts and citations. Use specific medical terms for best results.",
      inputSchema: {
        type: "object",
        properties: {
          query: {
            type: "string",
            minLength: 5,
            maxLength: 200,
            description: "Medical search terms (e.g., 'diabetes type 2 treatment')"
          },
          max_results: {
            type: "number",
            minimum: 1,
            maximum: 20,
            default: 5
          },
          publication_date_from: {
            type: "string",
            format: "date",
            description: "Only include papers published after this date"
          }
        },
        required: ["query"]
      }
    }
  ];
  
  async execute(toolName, args) {
    // Rate limiting (PubMed API has limits)
    await this.rateLimiter.acquire();
    
    try {
      // Call external API
      const response = await axios.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi', {
        params: {
          db: 'pubmed',
          term: args.query,
          retmax: args.max_results,
          retmode: 'json'
        },
        timeout: 5000  // 5 second timeout
      });
      
      return this.formatResults(response.data);
      
    } catch (error) {
      if (error.code === 'ECONNABORTED') {
        return {
          content: [{
            type: "text",
            text: "PubMed search timed out. Please try a more specific query."
          }],
          isError: true
        };
      }
      throw error;
    }
  }
}

// Server 3: Treatment MCP Server (Async Pattern)
class TreatmentMCPServer {
  tools = [
    {
      name: "generate_treatment_plan",
      description: "Generate AI-powered treatment plan based on patient data and diagnosis. **This tool takes 5-10 seconds to complete.** Use only when explicitly requested.",
      inputSchema: {
        type: "object",
        properties: {
          patient_id: { type: "string", pattern: "^P[0-9]{8}$" },
          diagnosis: {
            type: "string",
            minLength: 10,
            description: "Primary diagnosis (ICD-10 code or description)"
          },
          comorbidities: {
            type: "array",
            items: { type: "string" },
            description: "List of relevant comorbidities"
          },
          contraindications: {
            type: "array",
            items: { type: "string" },
            description: "Known drug allergies or contraindications"
          }
        },
        required: ["patient_id", "diagnosis"]
      }
    }
  ];
  
  async execute(toolName, args) {
    // Async execution pattern
    const jobId = generateJobId();
    
    // Start job (non-blocking)
    this.mlQueue.add({
      jobId,
      toolName,
      args,
      timestamp: Date.now()
    });
    
    // Poll for completion (with timeout)
    const startTime = Date.now();
    const timeout = 15000;  // 15 second max
    
    while (Date.now() - startTime < timeout) {
      const status = await this.jobStore.get(jobId);
      
      if (status?.complete) {
        return status.result;
      }
      
      // Progressive feedback
      if (Date.now() - startTime > 2000) {
        console.log(`Treatment plan generation in progress (${Math.floor((Date.now() - startTime) / 1000)}s elapsed)`);
      }
      
      await sleep(500);  // Poll every 500ms
    }
    
    throw new Error("Treatment plan generation timed out");
  }
}
```

**2. Caching Strategy:**

```typescript
class SmartCacheLayer {
  // Cache tier 1: In-memory (fastest)
  private memoryCache = new Map<string, CacheEntry>();
  
  // Cache tier 2: Redis (shared across instances)
  private redisCache: Redis;
  
  async get(key: string): Promise<any | null> {
    // Check memory first
    const memCached = this.memoryCache.get(key);
    if (memCached && !this.isExpired(memCached)) {
      return memCached.value;
    }
    
    // Check Redis
    const redisCached = await this.redisCache.get(key);
    if (redisCached) {
      const value = JSON.parse(redisCached);
      // Populate memory cache
      this.memoryCache.set(key, {
        value,
        expiry: Date.now() + 60000  // 1 min in memory
      });
      return value;
    }
    
    return null;
  }
  
  async setEx(key: string, ttl: number, value: any) {
    // Write to both tiers
    this.memoryCache.set(key, {
      value,
      expiry: Date.now() + Math.min(ttl, 60) * 1000
    });
    
    await this.redisCache.setEx(key, ttl, JSON.stringify(value));
  }
}

// Cache invalidation strategy
class CacheInvalidation {
  // Invalidate on patient record updates
  async onPatientUpdate(patientId: string) {
    // Invalidate all patient-related cache keys
    const pattern = `*patient*${patientId}*`;
    const keys = await this.redisCache.keys(pattern);
    
    if (keys.length > 0) {
      await this.redisCache.del(...keys);
      console.log(`Invalidated ${keys.length} cache entries for patient ${patientId}`);
    }
  }
  
  // Time-based expiry (TTL strategy)
  getCacheTTL(toolName: string): number {
    const ttlMap = {
      search_patient_by_id: 300,        // 5 min (patient data)
      search_patients_by_name: 600,     // 10 min (search results)
      search_medical_literature: 3600,  // 1 hour (literature stable)
      generate_treatment_plan: 86400    // 24 hours (expensive)
    };
    return ttlMap[toolName] || 300;
  }
}
```

**3. Handling Slow ML Model:**

```typescript
// Pattern 1: Async with Status Updates
class AsyncToolExecution {
  async executeWithProgress(toolName: string, args: any) {
    if (toolName === "generate_treatment_plan") {
      // Return immediately with job ID
      const jobId = await this.startBackgroundJob(toolName, args);
      
      return {
        content: [{
          type: "text",
          text: `Treatment plan generation started (Job ID: ${jobId}). This will take 5-10 seconds. I'll notify you when complete.`
        }],
        metadata: {
          jobId,
          status: "processing",
          estimatedTime: 7000  // 7 seconds estimate
        }
      };
    }
  }
  
  // Separate tool to check status
  async checkJobStatus(jobId: string) {
    const job = await this.jobStore.get(jobId);
    
    if (job.complete) {
      return {
        content: [{
          type: "text",
          text: job.result
        }],
        metadata: { status: "complete" }
      };
    } else {
      return {
        content: [{
          type: "text",
          text: `Treatment plan still generating... (${job.progress}% complete)`
        }],
        metadata: {
          status: "processing",
          progress: job.progress
        }
      };
    }
  }
}

// Pattern 2: Streaming Results (Future MCP Feature)
class StreamingToolExecution {
  async *executeStreaming(toolName: string, args: any) {
    if (toolName === "generate_treatment_plan") {
      yield { type: "status", text: "Analyzing patient data..." };
      await sleep(1000);
      
      yield { type: "status", text: "Reviewing medical guidelines..." };
      await sleep(2000);
      
      yield { type: "status", text: "Checking drug interactions..." };
      await sleep(2000);
      
      yield { type: "status", text: "Generating recommendations..." };
      const result = await this.mlModel.generate(args);
      
      yield { type: "result", text: result };
    }
  }
}
```

**4. Cost Optimization:**

```typescript
class CostOptimizer {
  // Batch similar requests
  private requestBatcher = new Map<string, Promise<any>>();
  
  async executeTool(toolName: string, args: any) {
    // If identical request is in-flight, reuse it
    const requestKey = `${toolName}:${JSON.stringify(args)}`;
    
    if (this.requestBatcher.has(requestKey)) {
      console.log(`Deduplicating request: ${requestKey}`);
      return await this.requestBatcher.get(requestKey);
    }
    
    // Execute and cache promise
    const promise = this.executeInternal(toolName, args);
    this.requestBatcher.set(requestKey, promise);
    
    // Clean up after completion
    promise.finally(() => {
      setTimeout(() => {
        this.requestBatcher.delete(requestKey);
      }, 1000);  // Keep for 1 second (catches rapid duplicates)
    });
    
    return await promise;
  }
  
  // Monitor costs
  async logCost(toolName: string, args: any, duration: number) {
    const costMap = {
      search_patient_by_id: 0.0001,           // $0.0001 per query
      search_medical_literature: 0.001,       // PubMed is free, but rate-limited
      generate_treatment_plan: 0.05,          // ML model is expensive
      schedule_appointment: 0.0005            // Calendar API cost
    };
    
    const cost = costMap[toolName] || 0;
    
    await this.metricsDb.insert({
      timestamp: Date.now(),
      toolName,
      cost,
      duration,
      cached: args._cached || false
    });
    
    // Alert if costs spike
    const dailyCost = await this.getDailyCost();
    if (dailyCost > 1000) {  // $1000/day threshold
      await this.alertOncall(`Daily MCP tool costs: $${dailyCost}`);
    }
  }
}
```

**Trade-offs Summary:**

| Aspect | Decision | Trade-off |
|--------|----------|-----------|
| **Caching** | 2-tier (memory + Redis) | 5min TTL | Freshness vs Performance |
| **ML Model** | Async execution | 15s timeout | UX vs Reliability |
| **Database** | Read replicas | Eventual consistency | Latency vs Data freshness |
| **Literature** | Rate limiting | 5 req/sec | Cost vs Availability |
| **Audit Logging** | Async to S3 | ~100ms latency | Compliance vs Performance |

**Key Insight:** In healthcare, **correctness > speed**. Better to have 200ms latency with proper validation than 50ms with errors.

---

## Step 7: Implementation - Building Your First MCP Server

Let's build a production-grade MCP server from scratch. Not a toy example - something you'd actually deploy.

### The Mental Model: MCP Server = HTTP API Server (minus HTTP)

You know how to build an Express.js API:
```typescript
app.post('/api/search', async (req, res) => {
  const { query } = req.body;
  const result = await searchDatabase(query);
  res.json(result);
});
```

An MCP server is **conceptually identical**, except:
- Instead of HTTP routes → You have **tools**
- Instead of `req.body` → You have **tool arguments** (JSON Schema validated)
- Instead of `res.json()` → You return **tool results**
- Instead of HTTP transport → You use **stdio or SSE**

The SDK abstracts the protocol layer (JSON-RPC 2.0), just like Express abstracts HTTP parsing.

### Real-World Example: GitHub MCP Server

Let's build a GitHub MCP server that:
1. Lists repositories for an organization
2. Creates issues
3. Gets PR details

**Why this example?** Because it demonstrates:
- External API integration (GitHub REST API)
- Authentication (GitHub token)
- Error handling (rate limits, 404s)
- Multiple tools with different schemas

### Step-by-Step Implementation

#### 1. Project Setup

```bash
mkdir github-mcp-server
cd github-mcp-server
npm init -y
npm install @modelcontextprotocol/sdk @octokit/rest dotenv
npm install -D @types/node typescript
```

**File structure:**
```
github-mcp-server/
├── src/
│   ├── index.ts          # Server entry point
│   ├── tools.ts          # Tool definitions
│   └── github-client.ts  # GitHub API wrapper
├── package.json
├── tsconfig.json
└── .env                  # GITHUB_TOKEN
```

#### 2. Core Server Implementation

```typescript
// src/index.ts
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { GitHubClient } from "./github-client.js";

// Initialize MCP Server
const server = new Server(
  {
    name: "github-mcp-server",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},  // Declare that this server provides tools
    },
  }
);

// Initialize GitHub client
const githubToken = process.env.GITHUB_TOKEN;
if (!githubToken) {
  throw new Error("GITHUB_TOKEN environment variable required");
}
const githubClient = new GitHubClient(githubToken);

// Tool 1: List Repositories
const listReposSchema = z.object({
  org: z.string().describe("GitHub organization name (e.g., 'microsoft')"),
  limit: z.number().min(1).max(100).default(10).describe("Number of repos to return"),
});

server.setRequestHandler("tools/list", async () => {
  return {
    tools: [
      {
        name: "list_repositories",
        description: "List repositories for a GitHub organization. Returns name, description, stars, and URL.",
        inputSchema: {
          type: "object",
          properties: {
            org: {
              type: "string",
              description: "GitHub organization name (e.g., 'microsoft')",
            },
            limit: {
              type: "number",
              minimum: 1,
              maximum: 100,
              default: 10,
              description: "Number of repos to return (1-100)",
            },
          },
          required: ["org"],
        },
      },
      {
        name: "create_issue",
        description: "Create a new issue in a GitHub repository. Returns issue number and URL.",
        inputSchema: {
          type: "object",
          properties: {
            owner: {
              type: "string",
              description: "Repository owner (user or org)",
            },
            repo: {
              type: "string",
              description: "Repository name",
            },
            title: {
              type: "string",
              description: "Issue title",
            },
            body: {
              type: "string",
              description: "Issue body (markdown supported)",
            },
            labels: {
              type: "array",
              items: { type: "string" },
              description: "Optional labels (e.g., ['bug', 'urgent'])",
            },
          },
          required: ["owner", "repo", "title", "body"],
        },
      },
      {
        name: "get_pull_request",
        description: "Get details of a pull request including status, reviews, and checks.",
        inputSchema: {
          type: "object",
          properties: {
            owner: { type: "string" },
            repo: { type: "string" },
            pr_number: { type: "number", description: "Pull request number" },
          },
          required: ["owner", "repo", "pr_number"],
        },
      },
    ],
  };
});

// Tool execution handler
server.setRequestHandler("tools/call", async (request) => {
  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      case "list_repositories": {
        const validated = listReposSchema.parse(args);
        const repos = await githubClient.listRepos(validated.org, validated.limit);
        
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(repos, null, 2),
            },
          ],
        };
      }

      case "create_issue": {
        // Validation + Execution
        const { owner, repo, title, body, labels } = args as any;
        const issue = await githubClient.createIssue(owner, repo, title, body, labels);
        
        return {
          content: [
            {
              type: "text",
              text: `✅ Issue created: #${issue.number}\nURL: ${issue.html_url}`,
            },
          ],
        };
      }

      case "get_pull_request": {
        const { owner, repo, pr_number } = args as any;
        const pr = await githubClient.getPullRequest(owner, repo, pr_number);
        
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                title: pr.title,
                state: pr.state,
                mergeable: pr.mergeable,
                reviews: pr.reviews,
                checks: pr.checks,
              }, null, 2),
            },
          ],
        };
      }

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (error: any) {
    // Error handling - return structured error to LLM
    return {
      content: [
        {
          type: "text",
          text: `❌ Error: ${error.message}`,
        },
      ],
      isError: true,
    };
  }
});

// Start server with stdio transport
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("GitHub MCP Server running on stdio");
}

main().catch((error) => {
  console.error("Server error:", error);
  process.exit(1);
});
```

#### 3. GitHub Client Wrapper

```typescript
// src/github-client.ts
import { Octokit } from "@octokit/rest";

export class GitHubClient {
  private octokit: Octokit;

  constructor(token: string) {
    this.octokit = new Octokit({ auth: token });
  }

  async listRepos(org: string, limit: number) {
    const { data } = await this.octokit.repos.listForOrg({
      org,
      per_page: limit,
      sort: "updated",
    });

    return data.map((repo) => ({
      name: repo.name,
      description: repo.description,
      stars: repo.stargazers_count,
      url: repo.html_url,
      language: repo.language,
    }));
  }

  async createIssue(
    owner: string,
    repo: string,
    title: string,
    body: string,
    labels?: string[]
  ) {
    const { data } = await this.octokit.issues.create({
      owner,
      repo,
      title,
      body,
      labels,
    });

    return {
      number: data.number,
      html_url: data.html_url,
    };
  }

  async getPullRequest(owner: string, repo: string, pr_number: number) {
    const [prData, reviews, checks] = await Promise.all([
      this.octokit.pulls.get({ owner, repo, pull_number: pr_number }),
      this.octokit.pulls.listReviews({ owner, repo, pull_number: pr_number }),
      this.octokit.checks.listForRef({
        owner,
        repo,
        ref: `pull/${pr_number}/head`,
      }),
    ]);

    return {
      title: prData.data.title,
      state: prData.data.state,
      mergeable: prData.data.mergeable,
      reviews: reviews.data.map((r) => ({
        user: r.user?.login,
        state: r.state,
      })),
      checks: checks.data.check_runs.map((c) => ({
        name: c.name,
        status: c.status,
        conclusion: c.conclusion,
      })),
    };
  }
}
```

#### 4. Build Configuration

```json
// tsconfig.json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "Node16",
    "moduleResolution": "Node16",
    "outDir": "./build",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "resolveJsonModule": true
  },
  "include": ["src/**/*"]
}
```

```json
// package.json (key parts)
{
  "name": "github-mcp-server",
  "version": "1.0.0",
  "type": "module",
  "bin": {
    "github-mcp-server": "./build/index.js"
  },
  "scripts": {
    "build": "tsc",
    "start": "node build/index.js"
  }
}
```

### Key Implementation Patterns

#### Pattern 1: Tool Registration (Discovery Phase)

The `tools/list` handler is called **once** when the MCP client connects. Think of it like Swagger/OpenAPI spec generation:

```typescript
server.setRequestHandler("tools/list", async () => {
  return {
    tools: [
      {
        name: "tool_name",           // Unique identifier
        description: "...",          // LLM reads this to decide when to use
        inputSchema: { /* JSON Schema */ },  // Validation contract
      },
    ],
  };
});
```

**Backend analogy:** This is like your API gateway returning available routes when you hit `/api/docs`.

#### Pattern 2: Input Validation

Two approaches:

**Approach A: JSON Schema only (simpler)**
```typescript
inputSchema: {
  type: "object",
  properties: {
    email: { type: "string", format: "email" },
    age: { type: "number", minimum: 0, maximum: 120 },
  },
  required: ["email"],
}
```

**Approach B: JSON Schema + Runtime validation (safer)**
```typescript
// Use Zod for TypeScript type safety + runtime validation
const schema = z.object({
  email: z.string().email(),
  age: z.number().min(0).max(120),
});

// In tools/call handler:
const validated = schema.parse(args);  // Throws if invalid
```

**Why both?** JSON Schema is for the LLM to understand the contract. Zod is for **runtime safety** because LLMs can hallucinate invalid JSON.

#### Pattern 3: Error Handling

```typescript
try {
  const result = await externalAPI.call();
  return {
    content: [{ type: "text", text: JSON.stringify(result) }],
  };
} catch (error: any) {
  // Return error to LLM (don't crash the server)
  return {
    content: [
      {
        type: "text",
        text: `Error: ${error.message}. ${error.response?.status === 404 ? "Resource not found." : "Please try again."}`,
      },
    ],
    isError: true,  // Optional flag
  };
}
```

**Why this matters:** If your tool crashes, the **entire MCP server dies**. The LLM can handle errors gracefully if you return them as structured responses.

#### Pattern 4: Response Content Types

MCP supports multiple content types:

```typescript
return {
  content: [
    {
      type: "text",
      text: "Simple text response",
    },
    {
      type: "image",
      data: base64ImageData,
      mimeType: "image/png",
    },
    {
      type: "resource",
      uri: "file:///path/to/file.txt",
      mimeType: "text/plain",
    },
  ],
};
```

**Use cases:**
- `text`: API responses, logs, structured data (JSON)
- `image`: Charts, diagrams, screenshots
- `resource`: File paths for the LLM to read later

### Testing Your MCP Server

Before integrating with an AI host, test the protocol manually:

```bash
# Build
npm run build

# Test discovery
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | GITHUB_TOKEN=your_token node build/index.js

# Test invocation
echo '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"list_repositories","arguments":{"org":"microsoft","limit":5}}}' | GITHUB_TOKEN=your_token node build/index.js
```

**Expected output:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "[{\"name\":\"vscode\",\"stars\":150000,...}]"
      }
    ]
  }
}
```

### Production Considerations

#### 1. Rate Limiting

```typescript
import { RateLimiter } from 'limiter';

const limiter = new RateLimiter({
  tokensPerInterval: 5000,  // GitHub rate limit
  interval: "hour",
});

async callGitHub() {
  await limiter.removeTokens(1);  // Block if rate limit exceeded
  return await this.octokit.request(...);
}
```

#### 2. Timeout Handling

```typescript
async function callWithTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) =>
      setTimeout(() => reject(new Error("Timeout")), timeoutMs)
    ),
  ]);
}

// Usage
const result = await callWithTimeout(
  githubClient.listRepos("microsoft", 100),
  5000  // 5 second timeout
);
```

#### 3. Logging & Observability

```typescript
// Log to stderr (stdout is reserved for JSON-RPC protocol)
console.error(`[${new Date().toISOString()}] Tool called: ${name}`);
console.error(`Arguments: ${JSON.stringify(args)}`);

// Production: Use structured logging
import winston from 'winston';
const logger = winston.createLogger({
  transports: [new winston.transports.File({ filename: 'mcp-server.log' })],
});
```

### Python Implementation (Alternative)

Same server in Python using the MCP SDK:

```python
# server.py
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from github import Github

# Initialize
server = Server("github-mcp-server")
github = Github(os.getenv("GITHUB_TOKEN"))

@server.list_tools()
async def list_tools():
    return [
        {
            "name": "list_repositories",
            "description": "List GitHub repositories for an organization",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "org": {"type": "string"},
                    "limit": {"type": "number", "default": 10},
                },
                "required": ["org"],
            },
        }
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "list_repositories":
        org = github.get_organization(arguments["org"])
        repos = org.get_repos()[:arguments.get("limit", 10)]
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps([{
                        "name": r.name,
                        "stars": r.stargazers_count,
                        "url": r.html_url,
                    } for r in repos]),
                }
            ]
        }

async def main():
    async with stdio_server() as streams:
        await server.run(streams[0], streams[1])

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Step 8: Integration - Connecting to AI Applications

You've built an MCP server. Now let's connect it to AI hosts.

### Integration Mental Model

Think of this like deploying a microservice:
- **MCP Server** = Your backend service
- **MCP Client** = Service mesh / API gateway
- **MCP Host** = The orchestrator (Kubernetes, but for LLMs)

The host (Cursor, Claude Desktop, custom app) manages:
- Server lifecycle (start/stop processes)
- Connection pooling
- Authentication
- Configuration

### Option 1: Cursor Integration

Cursor is an AI-powered IDE built on VS Code. It has native MCP support.

#### Configuration

Create `.cursor/mcp.json` in your project root:

```json
{
  "mcpServers": {
    "github": {
      "command": "node",
      "args": [
        "/absolute/path/to/github-mcp-server/build/index.js"
      ],
      "env": {
        "GITHUB_TOKEN": "ghp_your_token_here"
      }
    },
    "postgres": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-postgres",
        "postgresql://user:pass@localhost/mydb"
      ]
    },
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/you/Documents"
      ]
    }
  }
}
```

**Key points:**
- `command`: Executable to run (node, python, npx)
- `args`: Arguments passed to the command
- `env`: Environment variables (secrets, config)
- Server name (`github`, `postgres`) is arbitrary - just for your reference

#### How Cursor Uses Your Server

1. **Startup:** When you open Cursor, it spawns each MCP server as a child process using stdio transport
2. **Discovery:** Cursor calls `tools/list` on each server and merges all tools into a single catalog
3. **User Query:** When you type a prompt, the LLM decides which tools to use based on descriptions
4. **Invocation:** Cursor routes `tools/call` to the correct server
5. **Result:** LLM receives tool output and generates response

**Mermaid diagram:**

```mermaid
sequenceDiagram
    participant User
    participant Cursor as Cursor (MCP Host)
    participant GitHub as GitHub MCP Server
    participant Postgres as Postgres MCP Server
    
    User->>Cursor: "List Microsoft repos with >50k stars"
    
    Note over Cursor: LLM analyzes prompt,<br/>decides to use GitHub server
    
    Cursor->>GitHub: tools/call<br/>{name: "list_repositories",<br/>args: {org: "microsoft", limit: 100}}
    GitHub->>GitHub: Call GitHub API
    GitHub-->>Cursor: {content: [{type: "text",<br/>text: "[{name: 'vscode', stars: 150k}]"}]}
    
    Note over Cursor: LLM filters results<br/>(>50k stars)
    
    Cursor-->>User: "Found 3 repos:<br/>1. vscode (150k stars)<br/>2. TypeScript (95k stars)<br/>3. terminal (60k stars)"
```

### Option 2: Claude Desktop Integration

Claude Desktop is Anthropic's standalone app with MCP support.

#### Configuration

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "github": {
      "command": "node",
      "args": ["/path/to/github-mcp-server/build/index.js"],
      "env": {
        "GITHUB_TOKEN": "ghp_..."
      }
    }
  }
}
```

**Windows path:** `%APPDATA%\Claude\claude_desktop_config.json`

#### Debugging Claude Desktop Integration

If your server isn't showing up:

1. **Check logs:**
   ```bash
   # macOS
   tail -f ~/Library/Logs/Claude/mcp*.log
   
   # Windows
   type %APPDATA%\Claude\logs\mcp*.log
   ```

2. **Test server independently:**
   ```bash
   echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | node build/index.js
   ```

3. **Common issues:**
   - Incorrect path in config (must be absolute)
   - Missing environment variables
   - Node.js not in PATH
   - Server crashes on startup (check stderr logs)

### Option 3: Custom Application Integration

Build your own AI host using the MCP SDK.

#### TypeScript Example: CLI Chat Application

```typescript
// chat-app.ts
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import Anthropic from "@anthropic-ai/sdk";
import readline from "readline";

class AIChatApp {
  private mcpClient: Client;
  private anthropic: Anthropic;
  private conversationHistory: any[] = [];

  constructor() {
    this.anthropic = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY,
    });
    this.mcpClient = new Client(
      { name: "chat-app", version: "1.0.0" },
      { capabilities: {} }
    );
  }

  // Connect to MCP server
  async connectToMCPServer(command: string, args: string[]) {
    const transport = new StdioClientTransport({
      command,
      args,
      env: process.env,
    });

    await this.mcpClient.connect(transport);
    console.log("✅ Connected to MCP server");

    // Discover tools
    const tools = await this.mcpClient.listTools();
    console.log(`📦 Loaded ${tools.tools.length} tools:`, 
      tools.tools.map(t => t.name).join(", ")
    );
  }

  // Chat loop
  async chat() {
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });

    while (true) {
      const userMessage = await new Promise<string>((resolve) =>
        rl.question("You: ", resolve)
      );

      if (userMessage === "exit") break;

      // Add user message to history
      this.conversationHistory.push({
        role: "user",
        content: userMessage,
      });

      // Get available tools
      const toolsResponse = await this.mcpClient.listTools();
      const tools = toolsResponse.tools.map((tool) => ({
        name: tool.name,
        description: tool.description,
        input_schema: tool.inputSchema,
      }));

      // Call Claude with tool definitions
      let response = await this.anthropic.messages.create({
        model: "claude-3-5-sonnet-20241022",
        max_tokens: 4096,
        tools,
        messages: this.conversationHistory,
      });

      // Handle tool calls
      while (response.stop_reason === "tool_use") {
        const toolUse = response.content.find((c) => c.type === "tool_use");
        if (!toolUse || toolUse.type !== "tool_use") break;

        console.log(`🔧 Calling tool: ${toolUse.name}`);

        // Execute tool via MCP
        const toolResult = await this.mcpClient.callTool({
          name: toolUse.name,
          arguments: toolUse.input,
        });

        // Add assistant response + tool result to history
        this.conversationHistory.push({
          role: "assistant",
          content: response.content,
        });
        this.conversationHistory.push({
          role: "user",
          content: [
            {
              type: "tool_result",
              tool_use_id: toolUse.id,
              content: toolResult.content,
            },
          ],
        });

        // Continue conversation with tool result
        response = await this.anthropic.messages.create({
          model: "claude-3-5-sonnet-20241022",
          max_tokens: 4096,
          tools,
          messages: this.conversationHistory,
        });
      }

      // Extract final text response
      const textContent = response.content.find((c) => c.type === "text");
      if (textContent && textContent.type === "text") {
        console.log(`Claude: ${textContent.text}\n`);
        this.conversationHistory.push({
          role: "assistant",
          content: textContent.text,
        });
      }
    }
  }
}

// Run
const app = new AIChatApp();
await app.connectToMCPServer("node", [
  "/path/to/github-mcp-server/build/index.js",
]);
await app.chat();
```

**Run it:**
```bash
ANTHROPIC_API_KEY=sk-... GITHUB_TOKEN=ghp_... node chat-app.js
```

**Example conversation:**
```
You: List Microsoft's top 3 repos
🔧 Calling tool: list_repositories
Claude: Here are Microsoft's top 3 repos by stars:
1. vscode (150,234 stars) - Visual Studio Code
2. TypeScript (95,123 stars) - TypeScript language
3. terminal (60,456 stars) - Windows Terminal

You: Create an issue in the vscode repo about dark mode bug
🔧 Calling tool: create_issue
Claude: ✅ I created issue #87654 in microsoft/vscode:
"Dark mode theme not applying to sidebar"
URL: https://github.com/microsoft/vscode/issues/87654
```

### Option 4: Multi-Server Orchestration

In production, you'll use **multiple MCP servers**. Here's how to architect it:

#### Architecture Pattern

```mermaid
graph TD
    A[AI Application<br/>Custom Chat App] --> B[MCP Client Manager]
    
    B --> C[GitHub MCP Server<br/>stdio]
    B --> D[Database MCP Server<br/>stdio]
    B --> E[Analytics MCP Server<br/>SSE]
    B --> F[Document MCP Server<br/>SSE]
    
    C --> G[GitHub API]
    D --> H[PostgreSQL]
    E --> I[Snowflake]
    F --> J[S3 + Embeddings]
    
    style B fill:#e1f5ff
    style A fill:#fff4e6
```

#### Implementation: Multi-Server Manager

```typescript
// multi-server-manager.ts
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse.js";

interface ServerConfig {
  name: string;
  transport: "stdio" | "sse";
  command?: string;  // For stdio
  args?: string[];
  url?: string;      // For SSE
  env?: Record<string, string>;
}

class MCPServerManager {
  private clients: Map<string, Client> = new Map();
  private toolToServer: Map<string, string> = new Map();

  async connectServer(config: ServerConfig) {
    const client = new Client(
      { name: `manager-${config.name}`, version: "1.0.0" },
      { capabilities: {} }
    );

    // Create transport based on type
    let transport;
    if (config.transport === "stdio") {
      transport = new StdioClientTransport({
        command: config.command!,
        args: config.args || [],
        env: { ...process.env, ...config.env },
      });
    } else {
      transport = new SSEClientTransport(new URL(config.url!));
    }

    await client.connect(transport);
    this.clients.set(config.name, client);

    // Discover tools and map them to this server
    const tools = await client.listTools();
    tools.tools.forEach((tool) => {
      this.toolToServer.set(tool.name, config.name);
    });

    console.log(`✅ Connected to ${config.name} (${tools.tools.length} tools)`);
  }

  async getAllTools() {
    const allTools: any[] = [];
    for (const client of this.clients.values()) {
      const tools = await client.listTools();
      allTools.push(...tools.tools);
    }
    return allTools;
  }

  async callTool(name: string, args: any) {
    const serverName = this.toolToServer.get(name);
    if (!serverName) {
      throw new Error(`Tool not found: ${name}`);
    }

    const client = this.clients.get(serverName);
    if (!client) {
      throw new Error(`Server not connected: ${serverName}`);
    }

    return await client.callTool({ name, arguments: args });
  }

  async disconnectAll() {
    for (const [name, client] of this.clients) {
      await client.close();
      console.log(`Disconnected from ${name}`);
    }
  }
}

// Usage
const manager = new MCPServerManager();

await manager.connectServer({
  name: "github",
  transport: "stdio",
  command: "node",
  args: ["./servers/github/build/index.js"],
  env: { GITHUB_TOKEN: process.env.GITHUB_TOKEN },
});

await manager.connectServer({
  name: "database",
  transport: "stdio",
  command: "npx",
  args: ["-y", "@modelcontextprotocol/server-postgres", process.env.DATABASE_URL],
});

await manager.connectServer({
  name: "analytics",
  transport: "sse",
  url: "https://analytics-mcp.company.com",
});

// Now use tools from any server
const tools = await manager.getAllTools();
const result = await manager.callTool("list_repositories", { org: "microsoft" });
```

### Production Deployment Patterns

#### Pattern 1: stdio Servers (Local Tools)

**Use case:** IDE extensions, CLI tools, desktop apps

```yaml
# docker-compose.yml (if containerizing desktop app)
version: '3.8'
services:
  cursor-workspace:
    image: cursor-ai:latest
    volumes:
      - ./mcp-servers:/mcp-servers
      - ~/.cursor:/root/.cursor
    environment:
      - GITHUB_TOKEN=${GITHUB_TOKEN}
```

**Pros:**
- Fast (IPC, no network)
- Simple (no server management)

**Cons:**
- Not scalable (one process per connection)
- Local only

#### Pattern 2: SSE Servers (Remote Tools)

**Use case:** SaaS, multi-user, scalable systems

```yaml
# kubernetes/mcp-server-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: github-mcp-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: github-mcp
  template:
    metadata:
      labels:
        app: github-mcp
    spec:
      containers:
      - name: server
        image: my-org/github-mcp-server:latest
        ports:
        - containerPort: 3000
        env:
        - name: GITHUB_TOKEN
          valueFrom:
            secretKeyRef:
              name: github-creds
              key: token
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 512Mi
---
apiVersion: v1
kind: Service
metadata:
  name: github-mcp-service
spec:
  selector:
    app: github-mcp
  ports:
  - port: 80
    targetPort: 3000
  type: LoadBalancer
```

**Pros:**
- Scalable (load balancer + multiple replicas)
- Centralized (one deployment for all users)

**Cons:**
- Network latency
- More infrastructure

### Configuration Best Practices

#### 1. Environment-Specific Configs

```typescript
// config.ts
export const getConfig = () => {
  const env = process.env.NODE_ENV || 'development';
  
  const configs = {
    development: {
      mcpServers: {
        github: {
          command: 'node',
          args: ['./local-servers/github/build/index.js'],
        },
      },
    },
    production: {
      mcpServers: {
        github: {
          transport: 'sse',
          url: 'https://github-mcp.prod.company.com',
        },
      },
    },
  };
  
  return configs[env];
};
```

#### 2. Secrets Management

**Never** hardcode tokens in config files. Use:

```typescript
// For Cursor/Claude Desktop
{
  "mcpServers": {
    "github": {
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"  // Load from shell environment
      }
    }
  }
}
```

**For production apps:**
```typescript
import { SecretsManager } from 'aws-sdk';

const secrets = new SecretsManager({ region: 'us-east-1' });
const secret = await secrets.getSecretValue({ SecretId: 'mcp/github-token' }).promise();
const GITHUB_TOKEN = JSON.parse(secret.SecretString).token;
```

#### 3. Health Checks

```typescript
// Add to your MCP server
server.setRequestHandler("ping", async () => {
  return { status: "healthy" };
});

// In your host application
setInterval(async () => {
  try {
    await mcpClient.request({ method: "ping" });
  } catch (error) {
    console.error("MCP server unhealthy, reconnecting...");
    await reconnect();
  }
}, 30000);  // Every 30 seconds
```

---

## Checkpoint Question 4: E-Commerce Platform Integration

**Scenario:** You're a Senior Backend Engineer at a high-growth e-commerce company (Series C, 500 employees). The product team wants to add an AI assistant feature: "Users can chat with AI to track orders, get recommendations, and handle returns."

Your existing stack:
- **Orders Service:** Node.js + PostgreSQL (10M orders, 100k daily)
- **Recommendations Service:** Python + Redis + TensorFlow (ML model inference)
- **Returns Service:** Java Spring Boot + MySQL
- **Shipping Service:** Go + MongoDB (real-time tracking via FedEx/UPS APIs)

The AI team proposes: *"Build 4 MCP servers (one per service) and integrate with our chat interface."*

**Your Task:**

1. **Design the MCP architecture:** Which transport (stdio vs SSE) for each server? Why?
2. **Authentication strategy:** How do you ensure MCP tools respect user permissions? (e.g., User A can't see User B's orders)
3. **Performance optimization:** The recommendations ML model takes 800ms. How do you prevent chat latency?
4. **Error handling:** If the shipping service is down (FedEx API timeout), how should the AI respond?
5. **Draw a sequence diagram:** User asks "Where is my order #12345?" - show the full flow

**Constraints:**
- Chat interface must respond within 3 seconds
- Support 50k concurrent users
- Each user has 5-10 chat messages/session
- Orders/Returns services are internal (VPC), Shipping service is external (public API)

**Deliverables:**
- Architecture diagram (Mermaid)
- MCP server design decisions table
- Code snippet: Orders MCP Server with auth
- Failure handling strategy

---

## Step 9: Security & Authentication Patterns

MCP servers are **thin wrappers around existing APIs**. Security isn't about re-implementing authentication—it's about **securely delegating credentials**.

### The Mental Model: MCP Server = Reverse Proxy (Security Perspective)

Think of your MCP server like an **nginx reverse proxy**:
- It receives requests from the AI application
- It **forwards credentials** to the backend API
- It never stores or manages user data itself

The security question: *"How do I pass the right credentials to the right API?"*

### Anti-Pattern: Hardcoded Credentials

**DON'T DO THIS:**

```typescript
// ❌ BAD: Single credential for all users
server.setRequestHandler("tools/call", async (request) => {
  const { name, arguments: args } = request.params;
  
  if (name === "get_user_orders") {
    // Using server's own database credentials
    const orders = await db.query(
      "SELECT * FROM orders WHERE user_id = $1",
      [args.user_id]  // ❌ No verification - any user can see any orders!
    );
    return { content: [{ type: "text", text: JSON.stringify(orders) }] };
  }
});
```

**Why this is dangerous:**
- User A can query User B's orders by passing `user_id: "B"`
- No audit trail (all queries appear to be from the MCP server)
- Violates least-privilege principle

### Pattern 1: Request-Scoped Credentials (OAuth Token Delegation)

**The right way:** Pass user's OAuth token through the MCP request.

#### Architecture

```mermaid
sequenceDiagram
    participant User
    participant ChatUI as Chat UI
    participant MCP as MCP Host
    participant Server as MCP Server
    participant API as Backend API
    
    User->>ChatUI: Login (OAuth)
    ChatUI->>ChatUI: Store access_token
    
    User->>ChatUI: "Show my orders"
    ChatUI->>MCP: User message +<br/>user_token
    MCP->>Server: tools/call +<br/>Authorization header
    Server->>API: GET /orders<br/>Authorization: Bearer {user_token}
    API->>API: Validate token<br/>Extract user_id
    API-->>Server: Orders for user
    Server-->>MCP: Tool result
    MCP-->>ChatUI: AI response
    ChatUI-->>User: "You have 3 orders..."
```

#### Implementation

**1. MCP Server receives token in request context:**

```typescript
// orders-mcp-server.ts
import { Server } from "@modelcontextprotocol/sdk/server/index.js";

interface RequestContext {
  authorization?: string;  // User's OAuth token
  userId?: string;         // Extracted from token
}

const server = new Server(
  { name: "orders-mcp-server", version: "1.0.0" },
  { capabilities: { tools: {} } }
);

server.setRequestHandler("tools/call", async (request) => {
  const { name, arguments: args } = request.params;
  
  // Extract authorization header from request metadata
  const authHeader = request.params._meta?.authorization;
  if (!authHeader) {
    throw new Error("Unauthorized: No authorization header");
  }

  // Validate token and extract user_id
  const userId = await validateTokenAndGetUserId(authHeader);

  switch (name) {
    case "get_my_orders": {
      // Use user's token to call backend API
      const orders = await fetch("https://api.company.com/orders", {
        headers: {
          Authorization: authHeader,  // Forward user's token
        },
      }).then(r => r.json());

      // Backend API enforces authorization (returns only this user's orders)
      return {
        content: [{ type: "text", text: JSON.stringify(orders) }],
      };
    }
  }
});

async function validateTokenAndGetUserId(authHeader: string): Promise<string> {
  // Validate JWT or call auth service
  const response = await fetch("https://auth.company.com/validate", {
    headers: { Authorization: authHeader },
  });
  
  if (!response.ok) {
    throw new Error("Invalid token");
  }
  
  const { user_id } = await response.json();
  return user_id;
}
```

**2. MCP Host passes user context:**

```typescript
// In your AI application (MCP Host)
class AuthenticatedMCPClient {
  private mcpClient: Client;
  private userToken: string;

  constructor(mcpClient: Client, userToken: string) {
    this.mcpClient = mcpClient;
    this.userToken = userToken;
  }

  async callTool(name: string, args: any) {
    // Inject authorization into request
    return await this.mcpClient.callTool({
      name,
      arguments: {
        ...args,
        _meta: {
          authorization: `Bearer ${this.userToken}`,
        },
      },
    });
  }
}

// Usage
const userToken = req.headers.authorization; // From HTTP request
const authClient = new AuthenticatedMCPClient(mcpClient, userToken);
const result = await authClient.callTool("get_my_orders", {});
```

**Key points:**
- ✅ User's token is passed through MCP to backend API
- ✅ Backend API enforces authorization (not MCP server)
- ✅ Audit logs show actual user (not MCP server)
- ✅ Works with existing OAuth infrastructure

### Pattern 2: Service Account with Scoped Queries

For **internal tools** where users don't have direct API access (e.g., database queries).

#### Implementation

```typescript
// database-mcp-server.ts
server.setRequestHandler("tools/call", async (request) => {
  const { name, arguments: args } = request.params;
  
  // Extract verified user_id from token
  const userId = await validateTokenAndGetUserId(request.params._meta?.authorization);

  switch (name) {
    case "get_my_orders": {
      // MCP server uses its own DB credentials
      // BUT scopes the query to the authenticated user
      const orders = await db.query(
        "SELECT * FROM orders WHERE user_id = $1",  // ✅ Scoped to userId from token
        [userId]  // Not from args - from validated token!
      );

      return {
        content: [{ type: "text", text: JSON.stringify(orders) }],
      };
    }

    case "cancel_order": {
      const { order_id } = args;

      // Verify user owns this order before canceling
      const order = await db.queryOne(
        "SELECT user_id FROM orders WHERE id = $1",
        [order_id]
      );

      if (order.user_id !== userId) {
        throw new Error("Unauthorized: This order does not belong to you");
      }

      await db.query("UPDATE orders SET status = 'cancelled' WHERE id = $1", [order_id]);
      return {
        content: [{ type: "text", text: `Order ${order_id} cancelled` }],
      };
    }
  }
});
```

**Security checklist:**
- ✅ Token validation before ANY database operation
- ✅ User ID extracted from token (never from tool arguments)
- ✅ All queries scoped to authenticated user
- ✅ Ownership verification for mutations (cancel, update, delete)

### Pattern 3: API Key Management (SSE Servers)

For **remote MCP servers** (SSE transport), you need API-level authentication.

#### Multi-Layer Auth Architecture

```mermaid
graph TD
    A[AI Application] -->|1. User OAuth token| B[MCP Host]
    B -->|2. mTLS connection| C[MCP Server SSE]
    C -->|3. Validate API key| C
    C -->|4. User token| D[Backend APIs]
    
    style C fill:#fff4e6
    style B fill:#e1f5ff
```

**Layer 1: Client → MCP Server (API Key)**
```typescript
// MCP Host configuration
const transport = new SSEClientTransport(
  new URL("https://orders-mcp.company.com"),
  {
    headers: {
      "X-API-Key": process.env.MCP_API_KEY,  // Server-to-server auth
    },
  }
);
```

**Layer 2: MCP Server validates API key**
```typescript
// MCP Server (Express.js with SSE)
app.use((req, res, next) => {
  const apiKey = req.headers["x-api-key"];
  
  if (!apiKey || !isValidApiKey(apiKey)) {
    return res.status(401).json({ error: "Invalid API key" });
  }
  
  next();
});

function isValidApiKey(key: string): boolean {
  // Check against database or environment variable
  return key === process.env.ALLOWED_API_KEYS.split(",").includes(key);
}
```

**Layer 3: User authorization (same as Pattern 1)**
```typescript
// Tool execution still respects user context
server.setRequestHandler("tools/call", async (request) => {
  const userToken = request.params._meta?.authorization;
  const userId = await validateTokenAndGetUserId(userToken);
  
  // Proceed with user-scoped operations...
});
```

### Pattern 4: mTLS for High-Security Environments

For **financial services, healthcare, or regulated industries**.

#### Setup: Mutual TLS

```typescript
// MCP Server with mTLS
import https from "https";
import fs from "fs";

const options = {
  key: fs.readFileSync("server-key.pem"),
  cert: fs.readFileSync("server-cert.pem"),
  ca: fs.readFileSync("ca-cert.pem"),  // Certificate Authority
  requestCert: true,  // Require client certificate
  rejectUnauthorized: true,  // Reject invalid certs
};

const server = https.createServer(options, app);

app.use((req, res, next) => {
  const cert = req.socket.getPeerCertificate();
  
  if (!cert || !cert.subject) {
    return res.status(401).json({ error: "Client certificate required" });
  }
  
  // Verify client identity from certificate
  const clientId = cert.subject.CN;  // Common Name
  console.log(`Authenticated client: ${clientId}`);
  
  next();
});
```

**Client configuration:**
```typescript
// MCP Host with client certificate
const transport = new SSEClientTransport(
  new URL("https://orders-mcp.company.com"),
  {
    httpsAgent: new https.Agent({
      cert: fs.readFileSync("client-cert.pem"),
      key: fs.readFileSync("client-key.pem"),
      ca: fs.readFileSync("ca-cert.pem"),
    }),
  }
);
```

### Pattern 5: Secrets Management

**Never** hardcode secrets. Use environment variables or secret managers.

#### Development: Environment Variables

```typescript
// .env file (NEVER commit to git)
GITHUB_TOKEN=ghp_xxxxxxxxxxxxx
DATABASE_URL=postgresql://user:pass@localhost/db
OPENAI_API_KEY=sk-xxxxxxxx

// Load in server
import dotenv from "dotenv";
dotenv.config();

const githubToken = process.env.GITHUB_TOKEN;
```

#### Production: AWS Secrets Manager

```typescript
// secrets.ts
import { SecretsManagerClient, GetSecretValueCommand } from "@aws-sdk/client-secrets-manager";

const client = new SecretsManagerClient({ region: "us-east-1" });

export async function getSecret(secretName: string): Promise<string> {
  const response = await client.send(
    new GetSecretValueCommand({ SecretId: secretName })
  );
  
  if (!response.SecretString) {
    throw new Error(`Secret ${secretName} not found`);
  }
  
  return JSON.parse(response.SecretString).value;
}

// Usage in MCP server
const githubToken = await getSecret("mcp/github-token");
const dbPassword = await getSecret("mcp/database-password");
```

#### Production: Kubernetes Secrets

```yaml
# k8s-secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: mcp-secrets
type: Opaque
data:
  github-token: Z2hwX3h4eHh4eHh4  # Base64 encoded
  db-password: cGFzc3dvcmQxMjM=
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-server
spec:
  template:
    spec:
      containers:
      - name: server
        image: mcp-server:latest
        env:
        - name: GITHUB_TOKEN
          valueFrom:
            secretKeyRef:
              name: mcp-secrets
              key: github-token
        - name: DATABASE_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mcp-secrets
              key: db-password
```

### Pattern 6: Audit Logging

Log **every tool execution** for compliance and debugging.

```typescript
// audit-logger.ts
import winston from "winston";

const logger = winston.createLogger({
  format: winston.format.json(),
  transports: [
    new winston.transports.File({ filename: "mcp-audit.log" }),
  ],
});

export function logToolExecution(data: {
  userId: string;
  toolName: string;
  arguments: any;
  timestamp: Date;
  result: "success" | "error";
  duration: number;
}) {
  logger.info({
    event: "tool_execution",
    ...data,
  });
}

// Usage in server
server.setRequestHandler("tools/call", async (request) => {
  const startTime = Date.now();
  const userId = await extractUserId(request);
  const { name, arguments: args } = request.params;

  try {
    const result = await executeToolLogic(name, args);
    
    logToolExecution({
      userId,
      toolName: name,
      arguments: args,
      timestamp: new Date(),
      result: "success",
      duration: Date.now() - startTime,
    });

    return result;
  } catch (error: any) {
    logToolExecution({
      userId,
      toolName: name,
      arguments: args,
      timestamp: new Date(),
      result: "error",
      duration: Date.now() - startTime,
    });
    throw error;
  }
});
```

**Audit log output:**
```json
{
  "event": "tool_execution",
  "userId": "user_12345",
  "toolName": "cancel_order",
  "arguments": { "order_id": "ORD-789" },
  "timestamp": "2024-11-20T10:30:00Z",
  "result": "success",
  "duration": 245
}
```

### Security Checklist for Production

| Layer | Requirement | Implementation |
|-------|-------------|----------------|
| **Transport** | Encrypted connections | TLS 1.3 for SSE, stdio (local only) |
| **Authentication** | User identity verification | OAuth token validation |
| **Authorization** | User-scoped data access | Backend API enforces (not MCP) |
| **Secrets** | No hardcoded credentials | AWS Secrets Manager / K8s Secrets |
| **Audit** | Tool execution logging | Winston / CloudWatch Logs |
| **Network** | IP whitelisting (if applicable) | Security groups / firewall rules |
| **Rate Limiting** | Prevent abuse | Redis-based rate limiter |
| **Input Validation** | Prevent injection | JSON Schema + Zod validation |

### Common Security Mistakes

#### Mistake 1: Trusting tool arguments for user identity

```typescript
// ❌ BAD
server.setRequestHandler("tools/call", async (request) => {
  const { user_id } = request.params.arguments;  // ❌ User can fake this!
  const orders = await db.query("SELECT * FROM orders WHERE user_id = $1", [user_id]);
});

// ✅ GOOD
server.setRequestHandler("tools/call", async (request) => {
  const userId = await extractUserIdFromToken(request.params._meta.authorization);  // ✅ From validated token
  const orders = await db.query("SELECT * FROM orders WHERE user_id = $1", [userId]);
});
```

#### Mistake 2: Exposing internal IDs

```typescript
// ❌ BAD: Returning internal database IDs
{
  "order_id": 12345,  // Sequential ID - user can guess other IDs
  "internal_user_id": 67890
}

// ✅ GOOD: Use UUIDs or opaque tokens
{
  "order_id": "ord_a3f8b9c2d1e4",  // Unguessable
  "user_reference": "usr_x9y8z7"
}
```

#### Mistake 3: Insufficient error messages

```typescript
// ❌ BAD: Leaking system info
catch (error: any) {
  return { content: [{ type: "text", text: error.stack }] };  // ❌ Exposes file paths, DB structure
}

// ✅ GOOD: Safe error messages
catch (error: any) {
  console.error("Internal error:", error);  // Log full error server-side
  return {
    content: [{ type: "text", text: "Unable to process request. Please try again." }],
    isError: true,
  };
}
```

---

## Step 10: Observability & Control Flow

Two questions every backend engineer asks when debugging:
1. **"Why did the LLM call this tool?"** (Control flow)
2. **"Where is the bottleneck?"** (Performance)

### The Mental Model: MCP as Microservices Architecture

Think of your AI application as a **distributed system**:
- **MCP Host** = API Gateway
- **LLM** = Business logic service
- **MCP Servers** = Downstream services (database, cache, APIs)

You need the same observability as any microservices system: **logs, metrics, traces**.

### Part 1: Understanding Control Flow (How LLMs Decide)

#### The LLM Decision Process

```mermaid
sequenceDiagram
    participant User
    participant LLM
    participant Tools as Tool Catalog
    
    User->>LLM: "What's the weather in Tokyo?"
    
    Note over LLM: 1. Analyze intent<br/>"User wants weather data"
    
    LLM->>Tools: tools/list
    Tools-->>LLM: [get_weather, search_web,<br/>get_stock_price, ...]
    
    Note over LLM: 2. Match intent to tools<br/>Tool: get_weather<br/>Description: "Returns current weather..."<br/>Match score: HIGH
    
    Note over LLM: 3. Generate tool call<br/>{name: "get_weather",<br/>arguments: {city: "Tokyo"}}
    
    LLM->>Tools: tools/call + arguments
    Tools-->>LLM: {temperature: 15, conditions: "Cloudy"}
    
    Note over LLM: 4. Synthesize response<br/>"The weather in Tokyo is..."
    
    LLM-->>User: "The weather in Tokyo is 15°C and cloudy."
```

#### What Influences Tool Selection?

**1. Tool description quality:**

```typescript
// ❌ BAD: Vague description
{
  name: "search",
  description: "Searches for stuff",  // Too vague!
}

// ✅ GOOD: Specific description
{
  name: "search_github_repos",
  description: "Searches GitHub repositories by name, description, or topic. Returns repository names, star counts, and URLs. Use this when the user asks about finding GitHub projects, popular repos, or open source libraries.",
}
```

**2. Parameter names and descriptions:**

```typescript
// ❌ BAD: Unclear parameters
{
  name: "get_data",
  inputSchema: {
    type: "object",
    properties: {
      id: { type: "string" },  // What kind of ID?
    },
  },
}

// ✅ GOOD: Descriptive parameters
{
  name: "get_user_profile",
  inputSchema: {
    type: "object",
    properties: {
      user_id: {
        type: "string",
        description: "Unique user identifier (UUID format). Example: 'usr_a3f8b9c2'",
      },
      include_order_history: {
        type: "boolean",
        default: false,
        description: "Set to true to include the user's last 10 orders (slower query)",
      },
    },
  },
}
```

**3. Examples in descriptions:**

```typescript
{
  name: "calculate_shipping",
  description: `Calculates shipping cost for an order based on weight, destination, and shipping speed. 
  
Examples:
- For a 2kg package to New York with standard shipping: $8.50
- For a 5kg package to California with express: $24.99

Use this tool when the user asks "How much will shipping cost?" or "Calculate shipping for..."`
}
```

#### Debugging: Why didn't the LLM call my tool?

**Common reasons:**

1. **Tool description doesn't match user intent:**
   ```typescript
   // User: "Find Microsoft repos"
   // Tool description: "Searches GitHub organizations"  // ❌ Doesn't mention "repos"
   ```

2. **Too many similar tools (confusion):**
   ```typescript
   // Tools: search_repos, find_repositories, get_github_projects
   // LLM gets confused which one to use
   // Solution: Consolidate into ONE tool
   ```

3. **Missing keywords in description:**
   ```typescript
   // User: "What's the weather forecast?"
   // Tool description: "Returns current weather"  // ❌ Doesn't mention "forecast"
   ```

**Debugging technique: Prompt inspection**

```typescript
// Log the full prompt sent to LLM
const response = await anthropic.messages.create({
  model: "claude-3-5-sonnet-20241022",
  messages: conversationHistory,
  tools: toolDefinitions,
});

console.log("Tools provided to LLM:", JSON.stringify(toolDefinitions, null, 2));
console.log("LLM response:", JSON.stringify(response, null, 2));

if (response.stop_reason !== "tool_use") {
  console.warn("LLM did NOT call any tools. Reason:", response.stop_reason);
}
```

### Part 2: Distributed Tracing with OpenTelemetry

Instrument your MCP infrastructure like any microservices system.

#### Architecture

```mermaid
graph TD
    A[AI Application] -->|Trace ID: abc123| B[MCP Host]
    B -->|Span: tool_call| C[MCP Server 1]
    B -->|Span: tool_call| D[MCP Server 2]
    C -->|Span: db_query| E[PostgreSQL]
    D -->|Span: http_request| F[GitHub API]
    
    B --> G[OpenTelemetry Collector]
    C --> G
    D --> G
    G --> H[Jaeger/Tempo]
    
    style G fill:#fff4e6
    style H fill:#e1f5ff
```

#### Implementation: Instrumenting MCP Server

```typescript
// instrumentation.ts
import { NodeSDK } from "@opentelemetry/sdk-node";
import { getNodeAutoInstrumentations } from "@opentelemetry/auto-instrumentations-node";
import { JaegerExporter } from "@opentelemetry/exporter-jaeger";

const sdk = new NodeSDK({
  traceExporter: new JaegerExporter({
    endpoint: "http://localhost:14268/api/traces",
  }),
  instrumentations: [getNodeAutoInstrumentations()],
  serviceName: "github-mcp-server",
});

sdk.start();

// Graceful shutdown
process.on("SIGTERM", () => {
  sdk.shutdown().then(() => process.exit(0));
});
```

```typescript
// mcp-server.ts (with tracing)
import { trace, context, SpanStatusCode } from "@opentelemetry/api";

const tracer = trace.getTracer("mcp-server");

server.setRequestHandler("tools/call", async (request) => {
  const { name, arguments: args } = request.params;

  // Create span for this tool execution
  return await tracer.startActiveSpan(`tool.${name}`, async (span) => {
    span.setAttribute("tool.name", name);
    span.setAttribute("tool.arguments", JSON.stringify(args));

    try {
      const result = await executeToolLogic(name, args);
      
      span.setStatus({ code: SpanStatusCode.OK });
      span.setAttribute("tool.result.size", JSON.stringify(result).length);
      
      return result;
    } catch (error: any) {
      span.setStatus({ code: SpanStatusCode.ERROR, message: error.message });
      span.recordException(error);
      throw error;
    } finally {
      span.end();
    }
  });
});

async function executeToolLogic(name: string, args: any) {
  const span = trace.getActiveSpan();

  switch (name) {
    case "list_repositories": {
      // Create child span for GitHub API call
      return await tracer.startActiveSpan("github.api.list_repos", async (apiSpan) => {
        apiSpan.setAttribute("github.org", args.org);
        apiSpan.setAttribute("github.limit", args.limit);

        const startTime = Date.now();
        const repos = await githubClient.listRepos(args.org, args.limit);
        
        apiSpan.setAttribute("github.repos_returned", repos.length);
        apiSpan.setAttribute("duration_ms", Date.now() - startTime);
        apiSpan.end();

        return { content: [{ type: "text", text: JSON.stringify(repos) }] };
      });
    }
  }
}
```

#### Trace Visualization in Jaeger

```
Trace: User query "List Microsoft repos"
├─ [AI Application] chat.process_message (2.4s)
│  ├─ [LLM] anthropic.messages.create (1.8s)
│  ├─ [MCP Host] mcp.call_tool (580ms)
│  │  └─ [MCP Server] tool.list_repositories (570ms)
│  │     ├─ [GitHub API] github.api.list_repos (520ms)  ← Bottleneck!
│  │     └─ [Serialization] json.stringify (50ms)
│  └─ [LLM] anthropic.messages.create (200ms)
```

**Insights from traces:**
- 85% of time spent in GitHub API call
- Opportunity: Cache GitHub responses for 5 minutes
- Expected improvement: 2.4s → 0.7s (70% faster)

### Part 3: Metrics & Monitoring

#### Key Metrics to Track

```typescript
// metrics.ts
import { Counter, Histogram, Registry } from "prom-client";

const register = new Registry();

// Tool invocation counter
const toolInvocations = new Counter({
  name: "mcp_tool_invocations_total",
  help: "Total number of tool invocations",
  labelNames: ["tool_name", "status"],
  registers: [register],
});

// Tool execution duration
const toolDuration = new Histogram({
  name: "mcp_tool_duration_seconds",
  help: "Tool execution duration in seconds",
  labelNames: ["tool_name"],
  buckets: [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
  registers: [register],
});

// Error counter
const toolErrors = new Counter({
  name: "mcp_tool_errors_total",
  help: "Total number of tool execution errors",
  labelNames: ["tool_name", "error_type"],
  registers: [register],
});

// Export metrics endpoint
import express from "express";
const metricsApp = express();
metricsApp.get("/metrics", async (req, res) => {
  res.set("Content-Type", register.contentType);
  res.end(await register.metrics());
});
metricsApp.listen(9090);

// Usage in server
server.setRequestHandler("tools/call", async (request) => {
  const { name, arguments: args } = request.params;
  const startTime = Date.now();

  try {
    const result = await executeToolLogic(name, args);
    
    toolInvocations.inc({ tool_name: name, status: "success" });
    toolDuration.observe({ tool_name: name }, (Date.now() - startTime) / 1000);
    
    return result;
  } catch (error: any) {
    toolInvocations.inc({ tool_name: name, status: "error" });
    toolErrors.inc({ tool_name: name, error_type: error.constructor.name });
    throw error;
  }
});
```

#### Prometheus Queries

```promql
# Tool call rate (requests per second)
rate(mcp_tool_invocations_total[5m])

# P95 latency
histogram_quantile(0.95, mcp_tool_duration_seconds_bucket)

# Error rate percentage
sum(rate(mcp_tool_invocations_total{status="error"}[5m])) 
/ 
sum(rate(mcp_tool_invocations_total[5m])) * 100

# Top 5 slowest tools
topk(5, avg(mcp_tool_duration_seconds) by (tool_name))
```

#### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "MCP Server Monitoring",
    "panels": [
      {
        "title": "Tool Invocations/sec",
        "targets": [
          {
            "expr": "sum(rate(mcp_tool_invocations_total[5m])) by (tool_name)"
          }
        ]
      },
      {
        "title": "P95 Latency by Tool",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(mcp_tool_duration_seconds_bucket[5m])) by (tool_name, le))"
          }
        ]
      },
      {
        "title": "Error Rate %",
        "targets": [
          {
            "expr": "sum(rate(mcp_tool_invocations_total{status='error'}[5m])) / sum(rate(mcp_tool_invocations_total[5m])) * 100"
          }
        ]
      }
    ]
  }
}
```

### Part 4: Logging Best Practices

#### Structured Logging

```typescript
// logger.ts
import winston from "winston";

export const logger = winston.createLogger({
  level: "info",
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  defaultMeta: { service: "mcp-server" },
  transports: [
    new winston.transports.File({ filename: "error.log", level: "error" }),
    new winston.transports.File({ filename: "combined.log" }),
  ],
});

// Usage
server.setRequestHandler("tools/call", async (request) => {
  const { name, arguments: args } = request.params;
  const requestId = generateRequestId();

  logger.info("Tool invocation started", {
    request_id: requestId,
    tool_name: name,
    arguments: args,
  });

  try {
    const result = await executeToolLogic(name, args);
    
    logger.info("Tool invocation completed", {
      request_id: requestId,
      tool_name: name,
      result_size: JSON.stringify(result).length,
    });

    return result;
  } catch (error: any) {
    logger.error("Tool invocation failed", {
      request_id: requestId,
      tool_name: name,
      error: error.message,
      stack: error.stack,
    });
    throw error;
  }
});
```

**Log output:**
```json
{
  "timestamp": "2024-11-20T10:30:00.123Z",
  "level": "info",
  "message": "Tool invocation started",
  "service": "mcp-server",
  "request_id": "req_a3f8b9c2",
  "tool_name": "list_repositories",
  "arguments": { "org": "microsoft", "limit": 10 }
}
```

### Part 5: Debugging Complex Flows

#### Scenario: User query doesn't return expected results

**Step-by-step debugging:**

1. **Check if LLM called the right tool:**
   ```typescript
   console.log("LLM response:", JSON.stringify(llmResponse, null, 2));
   // Look for: stop_reason === "tool_use"
   // Check: content[0].name === expected_tool_name
   ```

2. **Verify tool arguments:**
   ```typescript
   console.log("Tool call:", {
     name: toolUse.name,
     arguments: toolUse.input,
   });
   // Check: Are arguments correct?
   // Common issue: LLM hallucinates invalid format
   ```

3. **Inspect tool execution:**
   ```typescript
   const result = await mcpClient.callTool({ name, arguments });
   console.log("Tool result:", result);
   // Check: Is the data structure correct?
   // Common issue: Tool returns error but formatted as success
   ```

4. **Trace through MCP server:**
   ```typescript
   // In MCP server, add detailed logging
   server.setRequestHandler("tools/call", async (request) => {
     console.error("Received:", JSON.stringify(request.params, null, 2));
     const result = await executeToolLogic(...);
     console.error("Returning:", JSON.stringify(result, null, 2));
     return result;
   });
   ```

5. **Check backend API:**
   ```bash
   # Test API directly
   curl -H "Authorization: Bearer $TOKEN" https://api.company.com/orders
   ```

#### Common Issues & Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **LLM doesn't call tool** | Returns text response instead of tool call | Improve tool description with keywords |
| **LLM calls wrong tool** | Calls `search_web` instead of `search_repos` | Make tool names more specific |
| **Invalid arguments** | Tool receives `{"city": "tokyo"}` instead of `{"city": "Tokyo"}` | Add examples in JSON Schema description |
| **Tool returns error** | Result: `"Error: 404 Not Found"` | Check backend API, verify credentials |
| **Slow response** | Takes >5 seconds | Check traces, identify bottleneck, add caching |

### Part 6: Performance Optimization

#### Pattern 1: Response Streaming

For long-running tools, stream partial results:

```typescript
// streaming-tool.ts
server.setRequestHandler("tools/call", async (request) => {
  const { name, arguments: args } = request.params;

  if (name === "analyze_large_dataset") {
    // Return streaming response
    return {
      content: [
        {
          type: "text",
          text: "Processing...",
          isPartial: true,  // Indicates more data coming
        },
      ],
    };
    
    // Continue processing in background
    processDatasetInBackground(args.dataset_id, (progress) => {
      // Send progress updates (implementation-specific)
      sendProgressUpdate(`Analyzed ${progress.rows_processed} rows...`);
    });
  }
});
```

#### Pattern 2: Parallel Tool Execution

Let LLMs call multiple tools in parallel:

```typescript
// In your AI application
const toolCalls = llmResponse.content.filter(c => c.type === "tool_use");

// Execute all tool calls in parallel
const results = await Promise.all(
  toolCalls.map(async (toolUse) => {
    const result = await mcpClient.callTool({
      name: toolUse.name,
      arguments: toolUse.input,
    });
    return { tool_use_id: toolUse.id, result };
  })
);
```

**Example:**
```
User: "Compare weather in Tokyo, Paris, and New York"
LLM calls:
  - get_weather({city: "Tokyo"})      ↓
  - get_weather({city: "Paris"})      ↓  Parallel execution
  - get_weather({city: "New York"})   ↓
  
Total time: max(320ms, 280ms, 310ms) = 320ms (not 910ms!)
```

#### Pattern 3: Smart Caching

```typescript
import Redis from "ioredis";
const redis = new Redis();

async function getCachedOrFetch(key: string, fetchFn: () => Promise<any>, ttlSeconds: number) {
  // Check cache
  const cached = await redis.get(key);
  if (cached) {
    logger.info("Cache hit", { key });
    return JSON.parse(cached);
  }

  // Cache miss - fetch and store
  logger.info("Cache miss", { key });
  const data = await fetchFn();
  await redis.setex(key, ttlSeconds, JSON.stringify(data));
  return data;
}

// Usage in tool
server.setRequestHandler("tools/call", async (request) => {
  if (request.params.name === "get_user_profile") {
    const userId = request.params.arguments.user_id;
    
    return await getCachedOrFetch(
      `user:${userId}`,
      () => database.getUserProfile(userId),
      300  // 5 minute TTL
    );
  }
});
```

### Observability Checklist

| Category | Metric/Tool | Purpose |
|----------|-------------|---------|
| **Tracing** | OpenTelemetry + Jaeger | End-to-end request flow, latency analysis |
| **Metrics** | Prometheus + Grafana | Tool call rates, P95 latency, error rates |
| **Logging** | Winston + CloudWatch | Debugging, audit trail, error investigation |
| **Alerting** | PagerDuty / Opsgenie | Error rate >5%, P95 latency >2s |
| **Dashboards** | Grafana | Real-time monitoring, capacity planning |

---

## Checkpoint Question 5: Observability for Multi-Tenant SaaS

**Scenario:** You're a Senior SRE at a B2B SaaS company (1000 customers, 50k users). You've deployed MCP servers for your AI assistant feature. After 2 weeks in production:

**Production issues:**
1. **Slow queries:** 95th percentile latency is 4.2 seconds (SLA: <2s)
2. **Intermittent failures:** 2-3% of tool calls fail with "Timeout" errors
3. **Cost spike:** OpenAI API costs increased 3x (from $2k/day to $6k/day)
4. **Customer complaint:** "Acme Corp's AI assistant is slower than others"

**Your infrastructure:**
- 4 MCP servers (Database, CRM, Analytics, Documents)
- Each deployed on Kubernetes (3 replicas)
- Transport: SSE over HTTP
- Backend: PostgreSQL (read replicas), Redis, S3

**Your Task:**

1. **Observability setup:** What metrics/traces would help diagnose these issues? List specific Prometheus queries.
2. **Root cause analysis:** Given the symptoms, what are 3 likely root causes for EACH issue?
3. **Optimization strategy:** Propose architectural changes to fix latency and cost issues.
4. **Multi-tenancy:** How would you isolate Acme Corp's performance from other customers?
5. **Alerting rules:** Define 3 critical alerts with thresholds and actions.

**Deliverables:**
- Mermaid diagram: Instrumented architecture (where to add traces/metrics)
- Prometheus queries for debugging
- Code snippet: Implement per-tenant metrics
- Incident response runbook

---

## Step 11: Role-Specific Scenarios - Part 1

These scenarios map MCP concepts to real-world engineering challenges you'll face in your role.

---

### 🎯 Backend Engineer (3-8 years experience)

You build APIs, manage databases, and integrate third-party services. MCP is **just another integration point** in your service architecture.

#### Scenario 1: REST API Wrapper for AI Access

**Context:** Your e-commerce company has a legacy Order Management System (OMS) with a REST API. The product team wants the AI assistant to query orders, but the AI team doesn't want direct database access (security policy).

**Current API endpoints:**
```
GET  /api/v1/orders?user_id={id}         # List orders
GET  /api/v1/orders/{order_id}            # Get order details
POST /api/v1/orders/{order_id}/cancel     # Cancel order
POST /api/v1/orders/{order_id}/refund     # Initiate refund
```

**Requirements:**
1. Expose these endpoints as MCP tools
2. Respect rate limits (100 req/min per user)
3. Validate user permissions (users can only see their own orders)
4. Log all operations for compliance

**Your Task:** Design and implement the MCP server.

**Solution:**

```typescript
// orders-mcp-server.ts
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import axios from "axios";
import { RateLimiterMemory } from "rate-limiter-flexible";

// Rate limiter: 100 requests per minute per user
const rateLimiter = new RateLimiterMemory({
  points: 100,
  duration: 60,
});

// OMS API client
const omsClient = axios.create({
  baseURL: process.env.OMS_API_URL || "https://oms.company.internal",
  timeout: 5000,
});

const server = new Server(
  { name: "orders-mcp-server", version: "1.0.0" },
  { capabilities: { tools: {} } }
);

// Extract user ID from JWT token
async function authenticateUser(authHeader: string): Promise<string> {
  if (!authHeader?.startsWith("Bearer ")) {
    throw new Error("Missing or invalid authorization header");
  }

  const token = authHeader.substring(7);
  
  // Validate JWT (use your actual auth service)
  const response = await axios.post(
    "https://auth.company.internal/validate",
    {},
    { headers: { Authorization: `Bearer ${token}` } }
  );

  if (!response.data.valid) {
    throw new Error("Invalid token");
  }

  return response.data.user_id;
}

// Tool definitions
server.setRequestHandler("tools/list", async () => {
  return {
    tools: [
      {
        name: "list_my_orders",
        description:
          "List orders for the authenticated user. Returns order ID, status, total amount, and creation date. Use this when the user asks 'show my orders', 'what did I order', or 'order history'.",
        inputSchema: {
          type: "object",
          properties: {
            status: {
              type: "string",
              enum: ["pending", "shipped", "delivered", "cancelled"],
              description: "Filter by order status (optional)",
            },
            limit: {
              type: "number",
              minimum: 1,
              maximum: 50,
              default: 10,
              description: "Number of orders to return",
            },
          },
        },
      },
      {
        name: "get_order_details",
        description:
          "Get detailed information about a specific order including items, shipping address, and tracking number. Use this when the user asks about a specific order like 'where is my order #12345'.",
        inputSchema: {
          type: "object",
          properties: {
            order_id: {
              type: "string",
              description: "Order ID (e.g., 'ORD-12345')",
            },
          },
          required: ["order_id"],
        },
      },
      {
        name: "cancel_order",
        description:
          "Cancel an order. Only works for orders in 'pending' or 'processing' status. Returns confirmation or error message.",
        inputSchema: {
          type: "object",
          properties: {
            order_id: {
              type: "string",
              description: "Order ID to cancel",
            },
            reason: {
              type: "string",
              description: "Cancellation reason (optional)",
            },
          },
          required: ["order_id"],
        },
      },
      {
        name: "request_refund",
        description:
          "Initiate a refund for a delivered order. Returns refund request ID and expected processing time (5-7 business days).",
        inputSchema: {
          type: "object",
          properties: {
            order_id: {
              type: "string",
              description: "Order ID to refund",
            },
            items: {
              type: "array",
              items: { type: "string" },
              description: "Optional: Specific item IDs to refund (for partial refunds)",
            },
            reason: {
              type: "string",
              description: "Refund reason",
            },
          },
          required: ["order_id", "reason"],
        },
      },
    ],
  };
});

// Tool execution
server.setRequestHandler("tools/call", async (request) => {
  const { name, arguments: args } = request.params;

  // 1. Authenticate user
  const authHeader = request.params._meta?.authorization;
  if (!authHeader) {
    return {
      content: [{ type: "text", text: "❌ Authentication required" }],
      isError: true,
    };
  }

  let userId: string;
  try {
    userId = await authenticateUser(authHeader);
  } catch (error: any) {
    return {
      content: [{ type: "text", text: `❌ Authentication failed: ${error.message}` }],
      isError: true,
    };
  }

  // 2. Rate limiting
  try {
    await rateLimiter.consume(userId);
  } catch (error) {
    return {
      content: [
        {
          type: "text",
          text: "❌ Rate limit exceeded. Please try again in a minute.",
        },
      ],
      isError: true,
    };
  }

  // 3. Execute tool
  try {
    switch (name) {
      case "list_my_orders": {
        const { status, limit = 10 } = args as any;

        const response = await omsClient.get("/api/v1/orders", {
          params: { user_id: userId, status, limit },
          headers: { Authorization: authHeader },
        });

        const orders = response.data.orders.map((order: any) => ({
          order_id: order.id,
          status: order.status,
          total: `$${order.total_amount}`,
          created_at: order.created_at,
          items_count: order.items.length,
        }));

        // Log for compliance
        console.log(
          JSON.stringify({
            event: "tool_execution",
            user_id: userId,
            tool: "list_my_orders",
            timestamp: new Date().toISOString(),
            result_count: orders.length,
          })
        );

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(orders, null, 2),
            },
          ],
        };
      }

      case "get_order_details": {
        const { order_id } = args as any;

        const response = await omsClient.get(`/api/v1/orders/${order_id}`, {
          headers: { Authorization: authHeader },
        });

        const order = response.data;

        // Authorization check: user must own this order
        if (order.user_id !== userId) {
          return {
            content: [
              {
                type: "text",
                text: "❌ Unauthorized: This order does not belong to you",
              },
            ],
            isError: true,
          };
        }

        console.log(
          JSON.stringify({
            event: "tool_execution",
            user_id: userId,
            tool: "get_order_details",
            order_id,
            timestamp: new Date().toISOString(),
          })
        );

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(order, null, 2),
            },
          ],
        };
      }

      case "cancel_order": {
        const { order_id, reason } = args as any;

        // First, verify ownership
        const checkResponse = await omsClient.get(`/api/v1/orders/${order_id}`, {
          headers: { Authorization: authHeader },
        });

        if (checkResponse.data.user_id !== userId) {
          return {
            content: [{ type: "text", text: "❌ Unauthorized" }],
            isError: true,
          };
        }

        // Cancel order
        const response = await omsClient.post(
          `/api/v1/orders/${order_id}/cancel`,
          { reason },
          { headers: { Authorization: authHeader } }
        );

        console.log(
          JSON.stringify({
            event: "tool_execution",
            user_id: userId,
            tool: "cancel_order",
            order_id,
            reason,
            timestamp: new Date().toISOString(),
          })
        );

        return {
          content: [
            {
              type: "text",
              text: `✅ Order ${order_id} has been cancelled. ${response.data.message}`,
            },
          ],
        };
      }

      case "request_refund": {
        const { order_id, items, reason } = args as any;

        // Verify ownership
        const checkResponse = await omsClient.get(`/api/v1/orders/${order_id}`, {
          headers: { Authorization: authHeader },
        });

        if (checkResponse.data.user_id !== userId) {
          return {
            content: [{ type: "text", text: "❌ Unauthorized" }],
            isError: true,
          };
        }

        // Initiate refund
        const response = await omsClient.post(
          `/api/v1/orders/${order_id}/refund`,
          { items, reason },
          { headers: { Authorization: authHeader } }
        );

        console.log(
          JSON.stringify({
            event: "tool_execution",
            user_id: userId,
            tool: "request_refund",
            order_id,
            reason,
            timestamp: new Date().toISOString(),
          })
        );

        return {
          content: [
            {
              type: "text",
              text: `✅ Refund request created: ${response.data.refund_id}\nExpected processing time: 5-7 business days`,
            },
          ],
        };
      }

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (error: any) {
    console.error(
      JSON.stringify({
        event: "tool_error",
        user_id: userId,
        tool: name,
        error: error.message,
        timestamp: new Date().toISOString(),
      })
    );

    // Distinguish between different error types
    if (error.response?.status === 404) {
      return {
        content: [{ type: "text", text: "❌ Order not found" }],
        isError: true,
      };
    } else if (error.response?.status === 403) {
      return {
        content: [
          {
            type: "text",
            text: "❌ This operation is not allowed for this order",
          },
        ],
        isError: true,
      };
    } else if (error.code === "ECONNABORTED") {
      return {
        content: [{ type: "text", text: "❌ Request timeout. Please try again." }],
        isError: true,
      };
    } else {
      return {
        content: [
          {
            type: "text",
            text: "❌ An error occurred. Please try again later.",
          },
        ],
        isError: true,
      };
    }
  }
});

// Start server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Orders MCP Server running");
}

main();
```

**Key Backend Patterns:**

1. **Authentication delegation:** Pass user's token to backend API
2. **Rate limiting:** Per-user limits using `rate-limiter-flexible`
3. **Authorization:** Verify ownership before mutations
4. **Error handling:** Map HTTP errors to user-friendly messages
5. **Audit logging:** JSON structured logs for compliance

---

#### Scenario 2: Database Query Tool with Connection Pooling

**Context:** Your analytics team wants AI to query the data warehouse (PostgreSQL with 500GB data). Direct SQL access is too risky. You need to expose safe, parameterized queries as MCP tools.

**Requirements:**
1. Predefined queries only (no arbitrary SQL)
2. Connection pooling (max 10 connections)
3. Query timeout: 30 seconds
4. Read-only access

**Solution:**

```typescript
// analytics-mcp-server.ts
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { Pool } from "pg";

// Connection pool (reuses connections)
const pool = new Pool({
  host: process.env.DB_HOST,
  database: process.env.DB_NAME,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  max: 10, // Max 10 connections
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
});

// Predefined queries (parameterized to prevent SQL injection)
const SAFE_QUERIES = {
  revenue_by_month: `
    SELECT 
      DATE_TRUNC('month', order_date) as month,
      SUM(total_amount) as revenue,
      COUNT(*) as order_count
    FROM orders
    WHERE order_date >= $1 AND order_date <= $2
    GROUP BY month
    ORDER BY month DESC
  `,
  
  top_products: `
    SELECT 
      product_id,
      product_name,
      SUM(quantity) as total_sold,
      SUM(quantity * price) as revenue
    FROM order_items oi
    JOIN products p ON oi.product_id = p.id
    WHERE oi.created_at >= $1 AND oi.created_at <= $2
    GROUP BY product_id, product_name
    ORDER BY revenue DESC
    LIMIT $3
  `,
  
  user_lifetime_value: `
    SELECT 
      user_id,
      email,
      COUNT(DISTINCT order_id) as order_count,
      SUM(total_amount) as lifetime_value,
      MAX(order_date) as last_order_date
    FROM orders o
    JOIN users u ON o.user_id = u.id
    WHERE user_id = $1
    GROUP BY user_id, email
  `,
};

const server = new Server(
  { name: "analytics-mcp-server", version: "1.0.0" },
  { capabilities: { tools: {} } }
);

server.setRequestHandler("tools/list", async () => {
  return {
    tools: [
      {
        name: "get_revenue_by_month",
        description:
          "Get monthly revenue and order count for a date range. Use this when user asks about revenue trends, monthly performance, or sales over time.",
        inputSchema: {
          type: "object",
          properties: {
            start_date: {
              type: "string",
              format: "date",
              description: "Start date (YYYY-MM-DD)",
            },
            end_date: {
              type: "string",
              format: "date",
              description: "End date (YYYY-MM-DD)",
            },
          },
          required: ["start_date", "end_date"],
        },
      },
      {
        name: "get_top_products",
        description:
          "Get best-selling products for a date range. Returns product name, quantity sold, and revenue. Use when user asks about 'top products', 'best sellers', or 'popular items'.",
        inputSchema: {
          type: "object",
          properties: {
            start_date: { type: "string", format: "date" },
            end_date: { type: "string", format: "date" },
            limit: {
              type: "number",
              minimum: 1,
              maximum: 100,
              default: 10,
              description: "Number of products to return",
            },
          },
          required: ["start_date", "end_date"],
        },
      },
      {
        name: "get_user_lifetime_value",
        description:
          "Calculate lifetime value for a specific user. Returns total orders, total spend, and last order date. Use when asking about a specific customer's history.",
        inputSchema: {
          type: "object",
          properties: {
            user_id: {
              type: "string",
              description: "User ID (UUID format)",
            },
          },
          required: ["user_id"],
        },
      },
    ],
  };
});

server.setRequestHandler("tools/call", async (request) => {
  const { name, arguments: args } = request.params;

  const client = await pool.connect();

  try {
    // Set query timeout
    await client.query("SET statement_timeout = '30s'");
    // Ensure read-only mode
    await client.query("SET transaction_read_only = on");

    let result;

    switch (name) {
      case "get_revenue_by_month": {
        const { start_date, end_date } = args as any;
        result = await client.query(SAFE_QUERIES.revenue_by_month, [
          start_date,
          end_date,
        ]);
        break;
      }

      case "get_top_products": {
        const { start_date, end_date, limit = 10 } = args as any;
        result = await client.query(SAFE_QUERIES.top_products, [
          start_date,
          end_date,
          limit,
        ]);
        break;
      }

      case "get_user_lifetime_value": {
        const { user_id } = args as any;
        result = await client.query(SAFE_QUERIES.user_lifetime_value, [user_id]);
        break;
      }

      default:
        throw new Error(`Unknown tool: ${name}`);
    }

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify(result.rows, null, 2),
        },
      ],
    };
  } catch (error: any) {
    if (error.message.includes("timeout")) {
      return {
        content: [
          {
            type: "text",
            text: "❌ Query timeout (>30s). Try a smaller date range.",
          },
        ],
        isError: true,
      };
    }

    return {
      content: [{ type: "text", text: `❌ Database error: ${error.message}` }],
      isError: true,
    };
  } finally {
    client.release(); // Return connection to pool
  }
});

// Graceful shutdown
process.on("SIGTERM", async () => {
  await pool.end();
  process.exit(0);
});

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Analytics MCP Server running");
}

main();
```

**Key Backend Patterns:**
- ✅ Connection pooling (pg.Pool)
- ✅ Parameterized queries (SQL injection prevention)
- ✅ Query timeout enforcement
- ✅ Read-only transaction mode
- ✅ Graceful connection cleanup

---

#### Interview Question: Database Scaling for MCP

**Question:** Your MCP server is experiencing database connection exhaustion (max 100 connections, all in use). The AI application has 500 concurrent users, each making 2-3 tool calls per minute. How would you scale this?

**Answer:**

**1. Analysis:**
- 500 users × 2.5 calls/min = 1,250 calls/min = ~21 calls/sec
- If each query takes 200ms, you need: 21 × 0.2 = 4.2 connections on average
- But bursts (50 queries at once) cause exhaustion

**2. Solutions (in order of implementation complexity):**

**Option A: Increase pool size + Read replicas**
```typescript
const pool = new Pool({
  max: 20,  // Increase from 10
  host: process.env.DB_READ_REPLICA,  // Use read replica
});
```
- Pros: Simple, works for 2-3x scale
- Cons: Limited by database capacity

**Option B: Connection pooling proxy (PgBouncer)**
```yaml
# pgbouncer.ini
[databases]
analytics = host=db.internal.com port=5432 dbname=analytics

[pgbouncer]
pool_mode = transaction  # Connection shared between queries
max_client_conn = 500     # Support 500 clients
default_pool_size = 20    # Only 20 actual DB connections
```
- Pros: 500 clients → 20 DB connections (25x multiplier)
- Cons: Requires infrastructure change

**Option C: Query result caching**
```typescript
import Redis from "ioredis";
const redis = new Redis();

async function cachedQuery(key: string, queryFn: () => Promise<any>) {
  const cached = await redis.get(key);
  if (cached) return JSON.parse(cached);

  const result = await queryFn();
  await redis.setex(key, 300, JSON.stringify(result));  // 5min TTL
  return result;
}

// Usage
case "get_revenue_by_month": {
  const cacheKey = `revenue:${start_date}:${end_date}`;
  const result = await cachedQuery(cacheKey, async () => {
    return await client.query(SAFE_QUERIES.revenue_by_month, [start_date, end_date]);
  });
}
```
- Pros: Reduces DB load by 80-90%
- Cons: Stale data (acceptable for analytics)

**Option D: Async query with job queue**
```typescript
import Bull from "bull";
const queryQueue = new Bull("analytics-queries");

// Tool returns immediately with job ID
case "get_revenue_by_month": {
  const job = await queryQueue.add({ query: "revenue_by_month", args });
  return {
    content: [{
      type: "text",
      text: `Query started. Job ID: ${job.id}. Check status in 10 seconds.`
    }]
  };
}

// Worker processes queries in background
queryQueue.process(async (job) => {
  const result = await runQuery(job.data.query, job.data.args);
  await redis.set(`query_result:${job.id}`, JSON.stringify(result), "EX", 3600);
});
```
- Pros: Handles slow queries (>5s) gracefully
- Cons: Complex, requires polling mechanism

**Recommended approach:**
- Short term: Option A + Option C (read replicas + caching)
- Long term: Option B (PgBouncer) for connection efficiency
- For heavy queries: Option D (async processing)

---

### 🎯 Cloud Engineer (4-10 years experience)

You architect cloud infrastructure, manage deployments, and optimize costs. MCP servers are **stateless services** that need cloud-native deployment patterns.

#### Scenario 1: Multi-Region Deployment with Failover

**Context:** Your company operates globally. Users in APAC, EMEA, and Americas need low-latency access to MCP tools. You need to deploy MCP servers in multiple regions with automatic failover.

**Requirements:**
1. Deploy in 3 regions: us-east-1, eu-west-1, ap-southeast-1
2. Route users to nearest region (<100ms latency)
3. Automatic failover if a region goes down
4. Shared state (Redis) for caching across regions

**Architecture:**

```mermaid
graph TD
    subgraph "Global"
        A[CloudFront CDN]
        B[Route 53 Latency-Based Routing]
    end
    
    subgraph "us-east-1"
        C[ALB US]
        D[MCP Server Pods US]
        E[Redis US Primary]
    end
    
    subgraph "eu-west-1"
        F[ALB EU]
        G[MCP Server Pods EU]
        H[Redis EU Replica]
    end
    
    subgraph "ap-southeast-1"
        I[ALB APAC]
        J[MCP Server Pods APAC]
        K[Redis APAC Replica]
    end
    
    subgraph "Shared Services"
        L[RDS PostgreSQL Multi-AZ]
        M[S3 Global]
    end
    
    A --> B
    B -->|Latency| C
    B -->|Latency| F
    B -->|Latency| I
    
    C --> D
    F --> G
    I --> J
    
    D --> E
    G --> H
    J --> K
    
    E -.Replication.-> H
    E -.Replication.-> K
    
    D --> L
    G --> L
    J --> L
    
    D --> M
    G --> M
    J --> M
```

**Implementation:**

```yaml
# terraform/main.tf
provider "aws" {
  alias  = "us"
  region = "us-east-1"
}

provider "aws" {
  alias  = "eu"
  region = "eu-west-1"
}

provider "aws" {
  alias  = "apac"
  region = "ap-southeast-1"
}

# EKS Clusters in each region
module "eks_us" {
  source = "./modules/eks"
  providers = { aws = aws.us }
  region = "us-east-1"
  cluster_name = "mcp-cluster-us"
}

module "eks_eu" {
  source = "./modules/eks"
  providers = { aws = aws.eu }
  region = "eu-west-1"
  cluster_name = "mcp-cluster-eu"
}

module "eks_apac" {
  source = "./modules/eks"
  providers = { aws = aws.apac }
  region = "ap-southeast-1"
  cluster_name = "mcp-cluster-apac"
}

# Route 53 latency-based routing
resource "aws_route53_record" "mcp_server" {
  zone_id = aws_route53_zone.main.zone_id
  name    = "mcp.company.com"
  type    = "A"

  set_identifier = "us-east-1"
  latency_routing_policy {
    region = "us-east-1"
  }

  alias {
    name                   = module.alb_us.dns_name
    zone_id                = module.alb_us.zone_id
    evaluate_target_health = true
  }
}

resource "aws_route53_record" "mcp_server_eu" {
  zone_id = aws_route53_zone.main.zone_id
  name    = "mcp.company.com"
  type    = "A"

  set_identifier = "eu-west-1"
  latency_routing_policy {
    region = "eu-west-1"
  }

  alias {
    name                   = module.alb_eu.dns_name
    zone_id                = module.alb_eu.zone_id
    evaluate_target_health = true
  }
}

# Similar for ap-southeast-1...

# Redis Global Datastore (cross-region replication)
resource "aws_elasticache_global_replication_group" "mcp_cache" {
  global_replication_group_id_suffix = "mcp-cache"
  primary_replication_group_id       = aws_elasticache_replication_group.us.id
}

resource "aws_elasticache_replication_group" "us" {
  provider = aws.us
  replication_group_id = "mcp-cache-us"
  engine               = "redis"
  node_type            = "cache.r6g.large"
  num_cache_clusters   = 2
  automatic_failover_enabled = true
}

# RDS Multi-Region (read replicas)
resource "aws_db_instance" "primary" {
  provider = aws.us
  identifier = "mcp-db-primary"
  engine     = "postgres"
  instance_class = "db.r6g.xlarge"
  multi_az   = true
}

resource "aws_db_instance" "replica_eu" {
  provider = aws.eu
  identifier = "mcp-db-replica-eu"
  replicate_source_db = aws_db_instance.primary.arn
}
```

```yaml
# kubernetes/deployment.yaml (per region)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-server
  namespace: production
spec:
  replicas: 5
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2
      maxUnavailable: 0
  selector:
    matchLabels:
      app: mcp-server
  template:
    metadata:
      labels:
        app: mcp-server
    spec:
      affinity:
        podAntiAffinity:  # Spread across AZs
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - mcp-server
            topologyKey: topology.kubernetes.io/zone
      containers:
      - name: mcp-server
        image: company/mcp-server:v1.2.3
        ports:
        - containerPort: 3000
        env:
        - name: AWS_REGION
          value: "us-east-1"  # Set per region
        - name: REDIS_ENDPOINT
          valueFrom:
            configMapKeyRef:
              name: infrastructure-config
              key: redis_endpoint
        - name: DB_ENDPOINT
          valueFrom:
            secretKeyRef:
              name: database-credentials
              key: endpoint
        resources:
          requests:
            cpu: 500m
            memory: 512Mi
          limits:
            cpu: 2000m
            memory: 2Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 3000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: mcp-server
  namespace: production
spec:
  type: LoadBalancer
  selector:
    app: mcp-server
  ports:
  - port: 80
    targetPort: 3000
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mcp-server-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: mcp-server
  minReplicas: 5
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

**Health check endpoint:**

```typescript
// health.ts
import express from "express";

const app = express();

app.get("/health", async (req, res) => {
  // Liveness: Is the process alive?
  res.status(200).json({ status: "healthy" });
});

app.get("/ready", async (req, res) => {
  // Readiness: Can it serve traffic?
  try {
    await Promise.all([
      checkRedis(),
      checkDatabase(),
      checkS3(),
    ]);
    res.status(200).json({ status: "ready" });
  } catch (error: any) {
    res.status(503).json({ status: "not ready", error: error.message });
  }
});

async function checkRedis() {
  const result = await redis.ping();
  if (result !== "PONG") throw new Error("Redis unhealthy");
}

async function checkDatabase() {
  await pool.query("SELECT 1");
}

async function checkS3() {
  await s3Client.send(new HeadBucketCommand({ Bucket: "mcp-data" }));
}

app.listen(3000);
```

**Cost optimization:**

```hcl
# terraform/cost-optimization.tf

# Use Spot instances for non-critical workloads
resource "aws_eks_node_group" "spot" {
  cluster_name    = module.eks_us.cluster_name
  node_group_name = "mcp-spot-nodes"
  capacity_type   = "SPOT"

  scaling_config {
    desired_size = 3
    max_size     = 20
    min_size     = 2
  }

  instance_types = ["t3.large", "t3a.large", "t2.large"]  # Multiple types for availability
}

# Use Reserved Instances for steady-state capacity
# (Purchase via AWS Console for 30-60% savings)

# Schedule scaling for time zones
resource "kubernetes_cron_job_v1" "scale_down_us_night" {
  metadata {
    name = "scale-down-us-night"
  }
  spec {
    schedule = "0 2 * * *"  # 2 AM US time (low traffic)
    job_template {
      spec {
        template {
          spec {
            containers {
              name  = "kubectl"
              image = "bitnami/kubectl"
              command = ["kubectl", "scale", "deployment/mcp-server", "--replicas=2"]
            }
          }
        }
      }
    }
  }
}
```

**Monitoring & Alerts:**

```yaml
# prometheus/alerts.yaml
groups:
- name: mcp_server
  rules:
  - alert: HighLatency
    expr: histogram_quantile(0.95, rate(mcp_tool_duration_seconds_bucket[5m])) > 2
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "P95 latency >2s in {{ $labels.region }}"
      
  - alert: HighErrorRate
    expr: sum(rate(mcp_tool_invocations_total{status="error"}[5m])) / sum(rate(mcp_tool_invocations_total[5m])) > 0.05
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Error rate >5% in {{ $labels.region }}"
      
  - alert: RegionDown
    expr: up{job="mcp-server"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "MCP server down in {{ $labels.region }}"
```

---

#### Scenario 2: Cost Optimization for Token-Heavy Workload

**Context:** Your MCP server calls external AI APIs (OpenAI embeddings for document search). Monthly cost: $15k. You need to reduce this by 50% without degrading performance.

**Solution: Multi-Tier Caching + Request Deduplication**

```typescript
// cost-optimized-mcp-server.ts
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import Redis from "ioredis";
import crypto from "crypto";
import { OpenAI } from "openai";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const redis = new Redis(process.env.REDIS_URL);

// In-memory cache (L1)
const memoryCache = new Map<string, { value: any; expiry: number }>();

// Cost tracking
let dailyCost = 0;
const COST_PER_1K_TOKENS = 0.0001; // Example rate

async function getCachedEmbedding(text: string): Promise<number[]> {
  const cacheKey = `emb:${crypto.createHash("md5").update(text).digest("hex")}`;

  // L1: Memory cache (0ms)
  const memoryCached = memoryCache.get(cacheKey);
  if (memoryCached && memoryCached.expiry > Date.now()) {
    console.log("L1 cache hit");
    return memoryCached.value;
  }

  // L2: Redis cache (~2ms)
  const redisCached = await redis.get(cacheKey);
  if (redisCached) {
    console.log("L2 cache hit");
    const embedding = JSON.parse(redisCached);
    
    // Populate L1
    memoryCache.set(cacheKey, {
      value: embedding,
      expiry: Date.now() + 60000, // 1 minute in L1
    });
    
    return embedding;
  }

  // L3: Call OpenAI API (150ms + $$)
  console.log("Cache miss - calling OpenAI");
  const response = await openai.embeddings.create({
    model: "text-embedding-ada-002",
    input: text,
  });

  const embedding = response.data[0].embedding;

  // Track cost
  const tokens = response.usage.total_tokens;
  const cost = (tokens / 1000) * COST_PER_1K_TOKENS;
  dailyCost += cost;
  
  await redis.incrbyfloat("cost:daily", cost);
  console.log(`Cost: $${cost.toFixed(4)} (Daily: $${dailyCost.toFixed(2)})`);

  // Store in caches
  await redis.setex(cacheKey, 86400, JSON.stringify(embedding)); // 24h in Redis
  memoryCache.set(cacheKey, {
    value: embedding,
    expiry: Date.now() + 60000,
  });

  return embedding;
}

server.setRequestHandler("tools/call", async (request) => {
  if (request.params.name === "search_documents") {
    const { query } = request.params.arguments as any;

    // Get query embedding (cached)
    const queryEmbedding = await getCachedEmbedding(query);

    // Search similar documents
    const results = await searchSimilarDocuments(queryEmbedding);

    return {
      content: [{ type: "text", text: JSON.stringify(results) }],
    };
  }
});
```

**Cost impact:**
- Before: 1M queries/day × $0.015 = $15k/month
- After (90% cache hit rate): 100k API calls × $0.015 = $1.5k/month
- **Savings: $13.5k/month (90%)**

---

#### Interview Question: Disaster Recovery for Stateless MCP Servers

**Question:** Your MCP server infrastructure (3 regions, 50 pods per region) goes down in us-east-1 due to an AWS availability zone failure. What's your DR strategy?

**Answer:**

**1. Detection (<1 minute):**
- Route 53 health checks fail for us-east-1 ALB
- Prometheus alert: `up{job="mcp-server", region="us-east-1"} == 0`
- PagerDuty incident created automatically

**2. Automatic Failover (<2 minutes):**
- Route 53 stops routing to us-east-1 (unhealthy)
- Traffic redistributes to eu-west-1 and ap-southeast-1
- HPA scales up remaining regions: 5 → 15 pods (3x)

**3. Data Consistency:**
- Redis: Global Datastore automatically promotes eu-west-1 to primary
- PostgreSQL: RDS Multi-AZ already has standby in us-east-1b (different AZ)
- S3: No impact (region-independent)

**4. Recovery Steps:**
1. **Immediate:** Traffic served by 2 regions (user impact: higher latency for US users, but functional)
2. **15 minutes:** AWS resolves AZ issue, EKS nodes come back online
3. **30 minutes:** Pods restart, pass health checks, Route 53 adds us-east-1 back
4. **1 hour:** HPA scales down eu/apac back to normal (5 pods)

**RTO (Recovery Time Objective):** <5 minutes (users rerouted to healthy regions)
**RPO (Recovery Point Objective):** 0 (stateless servers, no data loss)

**Key design decisions:**
- ✅ Multi-region deployment (no single point of failure)
- ✅ Stateless servers (no data on pods)
- ✅ Health-check based routing (automatic failover)
- ✅ Over-provisioning (can handle 2x traffic per region)

---

### 🎯 DevOps Engineer (3-8 years experience)

You automate deployments, manage CI/CD pipelines, and ensure reliable releases. MCP servers need **production-grade deployment automation**.

#### Scenario 1: Zero-Downtime Deployment Pipeline

**Context:** You need to deploy MCP server updates multiple times per day without downtime. Current process: manual kubectl apply, 2-minute downtime per deploy.

**Requirements:**
1. Zero downtime during deployments
2. Automated testing before production deploy
3. Automatic rollback if errors spike
4. Deploy to staging → canary → production

**Solution:**

```yaml
# .github/workflows/deploy.yml
name: Deploy MCP Server

on:
  push:
    branches: [main]

env:
  ECR_REGISTRY: ${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.us-east-1.amazonaws.com
  IMAGE_NAME: mcp-server

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Login to ECR
        uses: aws-actions/amazon-ecr-login@v1
      
      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: |
            ${{ env.ECR_REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
            ${{ env.ECR_REGISTRY }}/${{ env.IMAGE_NAME }}:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max

  test:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Run integration tests
        run: |
          docker run --rm \
            -e GITHUB_TOKEN=${{ secrets.GITHUB_TOKEN }} \
            ${{ env.ECR_REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
            npm test
      
      - name: Protocol compliance test
        run: |
          echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | \
          docker run -i ${{ env.ECR_REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} | \
          jq -e '.result.tools | length > 0'

  deploy-staging:
    needs: test
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - uses: azure/setup-kubectl@v3
      
      - name: Configure kubectl
        run: |
          aws eks update-kubeconfig --name mcp-cluster-staging --region us-east-1
      
      - name: Deploy to staging
        run: |
          kubectl set image deployment/mcp-server \
            mcp-server=${{ env.ECR_REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
            -n staging
          
          kubectl rollout status deployment/mcp-server -n staging --timeout=5m
      
      - name: Run smoke tests
        run: |
          STAGING_URL=$(kubectl get svc mcp-server -n staging -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
          curl -f http://$STAGING_URL/health || exit 1

  deploy-production-canary:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production
    steps:
      - name: Configure kubectl
        run: |
          aws eks update-kubeconfig --name mcp-cluster-prod --region us-east-1
      
      - name: Deploy canary (10% traffic)
        run: |
          # Create canary deployment
          kubectl apply -f - <<EOF
          apiVersion: apps/v1
          kind: Deployment
          metadata:
            name: mcp-server-canary
            namespace: production
          spec:
            replicas: 1  # 10% of 10 total pods
            selector:
              matchLabels:
                app: mcp-server
                version: canary
            template:
              metadata:
                labels:
                  app: mcp-server
                  version: canary
              spec:
                containers:
                - name: mcp-server
                  image: ${{ env.ECR_REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          EOF
          
          kubectl rollout status deployment/mcp-server-canary -n production
      
      - name: Monitor canary (5 minutes)
        run: |
          sleep 300  # Wait 5 minutes
          
          # Check error rate
          ERROR_RATE=$(curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(mcp_tool_invocations_total{status='error',version='canary'}[5m]))/sum(rate(mcp_tool_invocations_total{version='canary'}[5m]))" | jq -r '.data.result[0].value[1]')
          
          if (( $(echo "$ERROR_RATE > 0.05" | bc -l) )); then
            echo "Canary error rate too high: $ERROR_RATE"
            kubectl delete deployment mcp-server-canary -n production
            exit 1
          fi
          
          echo "Canary healthy, proceeding to full rollout"

  deploy-production-full:
    needs: deploy-production-canary
    runs-on: ubuntu-latest
    environment: production
    steps:
      - name: Configure kubectl
        run: |
          aws eks update-kubeconfig --name mcp-cluster-prod --region us-east-1
      
      - name: Rolling update (main deployment)
        run: |
          kubectl set image deployment/mcp-server \
            mcp-server=${{ env.ECR_REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
            -n production
          
          kubectl rollout status deployment/mcp-server -n production --timeout=10m
      
      - name: Remove canary
        run: |
          kubectl delete deployment mcp-server-canary -n production
      
      - name: Verify deployment
        run: |
          kubectl get deployment mcp-server -n production
          kubectl get pods -n production -l app=mcp-server

  rollback-on-failure:
    needs: deploy-production-full
    if: failure()
    runs-on: ubuntu-latest
    steps:
      - name: Rollback
        run: |
          aws eks update-kubeconfig --name mcp-cluster-prod --region us-east-1
          kubectl rollout undo deployment/mcp-server -n production
          echo "Deployment rolled back due to failure"
```

**Kubernetes Deployment with Progressive Delivery:**

```yaml
# kubernetes/production/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-server
  namespace: production
spec:
  replicas: 10
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2        # Add 2 new pods at a time
      maxUnavailable: 0  # Never reduce capacity (zero downtime)
  selector:
    matchLabels:
      app: mcp-server
      version: stable
  template:
    metadata:
      labels:
        app: mcp-server
        version: stable
    spec:
      containers:
      - name: mcp-server
        image: company/mcp-server:latest
        ports:
        - containerPort: 3000
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10
          failureThreshold: 3  # Mark unhealthy after 3 failures
        readinessProbe:
          httpGet:
            path: /ready
            port: 3000
          initialDelaySeconds: 10
          periodSeconds: 5
          successThreshold: 2   # Must pass 2 checks before routing traffic
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 15"]  # Graceful shutdown
```

**Automatic rollback with Flagger:**

```yaml
# flagger-canary.yaml
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: mcp-server
  namespace: production
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: mcp-server
  service:
    port: 80
    targetPort: 3000
  analysis:
    interval: 1m
    threshold: 5     # Rollback after 5 failed checks
    maxWeight: 50    # Max 50% traffic to canary
    stepWeight: 10   # Increase by 10% each step
    metrics:
    - name: request-success-rate
      thresholdRange:
        min: 95      # >95% success rate required
      interval: 1m
    - name: request-duration
      thresholdRange:
        max: 2000    # <2s P99 latency required
      interval: 1m
    webhooks:
    - name: load-test
      url: http://flagger-loadtester/
      timeout: 5s
      metadata:
        type: cmd
        cmd: "hey -z 1m -q 10 -c 2 http://mcp-server-canary:80/health"
```

---

#### Scenario 2: Secret Rotation Without Downtime

**Context:** Your MCP server uses API keys (GitHub, OpenAI, AWS) that must be rotated every 90 days for security compliance. Current process: update secrets, restart all pods (5-minute downtime).

**Solution:**

```yaml
# kubernetes/secret-rotation.yaml
apiVersion: v1
kind: Secret
metadata:
  name: api-keys
  namespace: production
  annotations:
    rotator.io/rotation-schedule: "0 0 */90 * *"  # Every 90 days
type: Opaque
data:
  github-token: <base64>
  openai-api-key: <base64>
---
apiVersion: batch/v1
kind: CronJob
metadata:
  name: secret-rotator
  namespace: production
spec:
  schedule: "0 0 */90 * *"  # Every 90 days at midnight
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: secret-rotator
          containers:
          - name: rotator
            image: company/secret-rotator:latest
            env:
            - name: SECRETS_TO_ROTATE
              value: "github-token,openai-api-key"
            command:
            - /bin/bash
            - -c
            - |
              # 1. Generate new API keys
              NEW_GITHUB_TOKEN=$(curl -X POST https://api.github.com/user/tokens ...)
              NEW_OPENAI_KEY=$(curl -X POST https://api.openai.com/v1/keys ...)
              
              # 2. Update Kubernetes secret
              kubectl create secret generic api-keys-new \
                --from-literal=github-token=$NEW_GITHUB_TOKEN \
                --from-literal=openai-api-key=$NEW_OPENAI_KEY \
                -n production
              
              # 3. Rolling restart (zero downtime)
              kubectl patch deployment mcp-server -n production \
                -p '{"spec":{"template":{"spec":{"volumes":[{"name":"api-keys","secret":{"secretName":"api-keys-new"}}]}}}}'
              
              kubectl rollout status deployment/mcp-server -n production
              
              # 4. Verify new keys work
              sleep 60
              ERROR_RATE=$(curl prometheus:9090/api/v1/query?query=rate(mcp_tool_errors[5m]))
              if [ "$ERROR_RATE" -gt "0.01" ]; then
                echo "High error rate detected, rolling back"
                kubectl rollout undo deployment/mcp-server -n production
                exit 1
              fi
              
              # 5. Delete old secret
              kubectl delete secret api-keys -n production
              kubectl rename secret api-keys-new api-keys -n production
              
              # 6. Revoke old API keys
              curl -X DELETE https://api.github.com/tokens/$OLD_GITHUB_TOKEN
```

---

#### Interview Question: Debugging Failed Deployment

**Question:** You deployed a new MCP server version. After 10 minutes, error rate spiked from 0.5% to 15%. Kubernetes rolled back automatically. How do you debug?

**Answer:**

**1. Check rollback event:**
```bash
kubectl rollout history deployment/mcp-server -n production
kubectl describe deployment mcp-server -n production | grep -A 10 "Events"
```

**2. Compare current vs previous version:**
```bash
# Get image tags
OLD_IMAGE=$(kubectl rollout history deployment/mcp-server -n production --revision=5 | grep Image | awk '{print $2}')
NEW_IMAGE=$(kubectl rollout history deployment/mcp-server -n production --revision=6 | grep Image | awk '{print $2}')

echo "Rolled back from: $NEW_IMAGE"
echo "Back to: $OLD_IMAGE"

# Check git diff
git diff $(echo $OLD_IMAGE | cut -d: -f2) $(echo $NEW_IMAGE | cut -d: -f2)
```

**3. Check logs from failed pods:**
```bash
# Get pods from failed revision
kubectl logs -n production -l app=mcp-server,version=canary --tail=100 --timestamps
```

**4. Check Prometheus metrics:**
```promql
# Which tool caused errors?
topk(5, sum(rate(mcp_tool_invocations_total{status="error", version="canary"}[5m])) by (tool_name))

# What types of errors?
sum(rate(mcp_tool_errors_total{version="canary"}[5m])) by (error_type)
```

**5. Check for infrastructure changes:**
```bash
# Did any dependencies change?
kubectl get events -n production --sort-by='.lastTimestamp' | head -20

# Database connectivity?
kubectl exec -it <pod-name> -n production -- nc -zv postgres.internal.com 5432

# Redis connectivity?
kubectl exec -it <pod-name> -n production -- redis-cli -h redis.internal.com ping
```

**Common root causes:**
1. **Missing environment variable:** New code expects `NEW_FEATURE_FLAG=true`, not set
2. **Dependency version mismatch:** Updated `axios` from v0.27 to v1.0 (breaking change)
3. **External API changed:** GitHub API deprecated an endpoint
4. **Database migration not applied:** New code queries a column that doesn't exist yet
5. **Rate limit hit:** New version makes 10x more API calls

**Fix process:**
1. Identify root cause from logs/metrics
2. Fix locally, add integration test
3. Deploy to staging, verify fix
4. Re-deploy to production with fixed version

---

## Step 12: Role-Specific Scenarios - Part 2

Advanced scenarios for senior engineering leaders managing MCP infrastructure at scale.

---

### 🎯 Site Reliability Engineer (5-12 years experience)

You own production reliability, SLOs, incident response, and capacity planning. MCP servers are **critical infrastructure** that needs 99.9% uptime.

#### Scenario 1: SLO/SLI Definition and Enforcement

**Context:** Your company's AI assistant has 100k monthly active users. The product team wants "fast and reliable" tool execution. You need to define measurable SLOs and build enforcement mechanisms.

**Stakeholder requirements:**
- Product: "Users shouldn't wait more than 3 seconds"
- Engineering: "We need some error budget for deployments"
- Finance: "Keep infrastructure costs under $50k/month"

**Your Task:** Define SLIs, SLOs, error budgets, and implement monitoring.

**Solution:**

**1. SLI/SLO Definition:**

```yaml
# slo-definition.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mcp-slos
  namespace: production
data:
  slos.yaml: |
    service: mcp-tools
    slos:
      - name: availability
        description: "Percentage of successful tool calls"
        sli:
          metric: mcp_tool_invocations_total
          query: |
            sum(rate(mcp_tool_invocations_total{status="success"}[5m])) /
            sum(rate(mcp_tool_invocations_total[5m]))
        objective: 99.9%
        error_budget: 0.1%
        window: 30d
        
      - name: latency
        description: "95th percentile tool execution time"
        sli:
          metric: mcp_tool_duration_seconds
          query: |
            histogram_quantile(0.95,
              sum(rate(mcp_tool_duration_seconds_bucket[5m])) by (le)
            )
        objective: 2000ms
        window: 30d
        
      - name: data_freshness
        description: "Cache hit rate (data not older than 5min)"
        sli:
          metric: mcp_cache_hits_total
          query: |
            sum(rate(mcp_cache_hits_total[5m])) /
            sum(rate(mcp_cache_requests_total[5m]))
        objective: 85%
        window: 7d
```

**2. Error Budget Calculation:**

```typescript
// error-budget-calculator.ts
interface SLO {
  name: string;
  objective: number; // e.g., 99.9 for 99.9%
  windowDays: number;
}

interface ErrorBudget {
  totalRequests: number;
  allowedFailures: number;
  actualFailures: number;
  budgetRemaining: number;
  budgetRemainingPercent: number;
  daysUntilReset: number;
}

async function calculateErrorBudget(
  slo: SLO,
  prometheusUrl: string
): Promise<ErrorBudget> {
  // Query total requests in SLO window
  const totalQuery = `sum(increase(mcp_tool_invocations_total[${slo.windowDays}d]))`;
  const totalRequests = await queryPrometheus(prometheusUrl, totalQuery);

  // Calculate allowed failures based on SLO
  const successRate = slo.objective / 100;
  const errorRate = 1 - successRate;
  const allowedFailures = Math.floor(totalRequests * errorRate);

  // Query actual failures
  const failuresQuery = `sum(increase(mcp_tool_invocations_total{status="error"}[${slo.windowDays}d]))`;
  const actualFailures = await queryPrometheus(prometheusUrl, failuresQuery);

  // Calculate remaining budget
  const budgetRemaining = allowedFailures - actualFailures;
  const budgetRemainingPercent = (budgetRemaining / allowedFailures) * 100;

  return {
    totalRequests,
    allowedFailures,
    actualFailures,
    budgetRemaining,
    budgetRemainingPercent,
    daysUntilReset: calculateDaysUntilReset(slo.windowDays),
  };
}

async function queryPrometheus(url: string, query: string): Promise<number> {
  const response = await fetch(
    `${url}/api/v1/query?query=${encodeURIComponent(query)}`
  );
  const data = await response.json();
  return parseFloat(data.data.result[0].value[1]);
}

function calculateDaysUntilReset(windowDays: number): number {
  const now = new Date();
  const startOfMonth = new Date(now.getFullYear(), now.getMonth(), 1);
  const nextMonth = new Date(now.getFullYear(), now.getMonth() + 1, 1);
  return Math.ceil(
    (nextMonth.getTime() - now.getTime()) / (1000 * 60 * 60 * 24)
  );
}

// Example usage
const availabilitySLO: SLO = {
  name: "availability",
  objective: 99.9,
  windowDays: 30,
};

const budget = await calculateErrorBudget(
  availabilitySLO,
  "http://prometheus:9090"
);

console.log(`
Error Budget Status:
- Total requests (30d): ${budget.totalRequests.toLocaleString()}
- Allowed failures: ${budget.allowedFailures}
- Actual failures: ${budget.actualFailures}
- Budget remaining: ${budget.budgetRemaining} (${budget.budgetRemainingPercent.toFixed(1)}%)
- Days until reset: ${budget.daysUntilReset}
`);

// Example output:
// Error Budget Status:
// - Total requests (30d): 45,000,000
// - Allowed failures: 45,000 (0.1% of 45M)
// - Actual failures: 12,500
// - Budget remaining: 32,500 (72.2%)
// - Days until reset: 8
```

**3. Error Budget Policy (Change Management):**

```typescript
// deployment-gate.ts
async function canDeploy(service: string): Promise<{ allowed: boolean; reason: string }> {
  const slos = await getSLOs(service);
  
  for (const slo of slos) {
    const budget = await calculateErrorBudget(slo, PROMETHEUS_URL);
    
    // Policy: Block deploys if error budget < 20%
    if (budget.budgetRemainingPercent < 20) {
      return {
        allowed: false,
        reason: `Error budget for ${slo.name} is ${budget.budgetRemainingPercent.toFixed(1)}% (threshold: 20%). Focus on reliability before new features.`,
      };
    }
  }
  
  return { allowed: true, reason: "All SLOs within budget" };
}

// CI/CD integration
const deployCheck = await canDeploy("mcp-tools");

if (!deployCheck.allowed) {
  console.error(`❌ Deployment blocked: ${deployCheck.reason}`);
  process.exit(1);
}

console.log(`✅ Deployment approved: ${deployCheck.reason}`);
```

**4. SLO Dashboard (Grafana):**

```json
{
  "dashboard": {
    "title": "MCP SLO Dashboard",
    "panels": [
      {
        "title": "Availability SLO",
        "targets": [
          {
            "expr": "sum(rate(mcp_tool_invocations_total{status='success'}[5m])) / sum(rate(mcp_tool_invocations_total[5m])) * 100"
          }
        ],
        "thresholds": [
          { "value": 99.9, "color": "green" },
          { "value": 99.5, "color": "yellow" },
          { "value": 0, "color": "red" }
        ]
      },
      {
        "title": "Error Budget Remaining",
        "targets": [
          {
            "expr": "(45000 - sum(increase(mcp_tool_invocations_total{status='error'}[30d]))) / 45000 * 100"
          }
        ],
        "thresholds": [
          { "value": 50, "color": "green" },
          { "value": 20, "color": "yellow" },
          { "value": 0, "color": "red" }
        ]
      },
      {
        "title": "P95 Latency vs SLO",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(mcp_tool_duration_seconds_bucket[5m])) by (le)) * 1000",
            "legendFormat": "P95 Latency"
          },
          {
            "expr": "2000",
            "legendFormat": "SLO Threshold"
          }
        ]
      }
    ]
  }
}
```

**5. Alerting Rules:**

```yaml
# prometheus/slo-alerts.yaml
groups:
- name: slo_alerts
  rules:
  # Fast burn (1h) - page immediately
  - alert: HighErrorRateFastBurn
    expr: |
      (
        sum(rate(mcp_tool_invocations_total{status="error"}[1h]))
        /
        sum(rate(mcp_tool_invocations_total[1h]))
      ) > (14.4 * 0.001)  # 14.4x error budget burn rate
    for: 5m
    labels:
      severity: critical
      slo: availability
    annotations:
      summary: "Burning through error budget 14.4x faster than normal"
      description: "At this rate, entire 30-day error budget will be exhausted in 2 days"
      
  # Slow burn (6h) - ticket for investigation
  - alert: HighErrorRateSlowBurn
    expr: |
      (
        sum(rate(mcp_tool_invocations_total{status="error"}[6h]))
        /
        sum(rate(mcp_tool_invocations_total[6h]))
      ) > (6 * 0.001)  # 6x error budget burn rate
    for: 30m
    labels:
      severity: warning
      slo: availability
    annotations:
      summary: "Elevated error rate for 6 hours"
      description: "Error budget burning 6x faster than sustainable rate"
      
  # Latency SLO breach
  - alert: LatencySLOBreach
    expr: |
      histogram_quantile(0.95,
        sum(rate(mcp_tool_duration_seconds_bucket[5m])) by (le)
      ) > 2
    for: 10m
    labels:
      severity: warning
      slo: latency
    annotations:
      summary: "P95 latency exceeds 2s SLO"
      description: "95th percentile latency: {{ $value }}s"
```

---

#### Scenario 2: Incident Response Runbook

**Context:** It's 3 AM. PagerDuty wakes you up: "MCP server error rate is 25% (SLO: 0.1%)". You have 15 minutes to mitigate before users notice widespread issues.

**Your Task:** Create incident response runbook and automation.

**Solution:**

**1. Incident Runbook:**

```markdown
# Incident Runbook: MCP Server High Error Rate

## Severity Matrix
| Error Rate | Severity | Response Time | Actions |
|------------|----------|---------------|---------|
| 1-5% | SEV3 | 1 hour | Investigate, create ticket |
| 5-15% | SEV2 | 15 minutes | Page on-call, start mitigation |
| >15% | SEV1 | Immediate | Page entire team, executive notification |

## Response Checklist

### 1. Acknowledge & Assess (2 minutes)
- [ ] Acknowledge PagerDuty alert
- [ ] Check error budget: `kubectl exec -it prometheus-0 -- curl http://localhost:9090/api/v1/query?query=error_budget_remaining`
- [ ] Identify scope: All tools or specific tool?
  ```bash
  kubectl logs -n production -l app=mcp-server --tail=100 | grep ERROR | cut -d' ' -f5 | sort | uniq -c | sort -rn
  ```

### 2. Immediate Mitigation (5 minutes)
Choose fastest mitigation:

**Option A: Rollback (if recent deploy)**
```bash
# Check recent deploys
kubectl rollout history deployment/mcp-server -n production | tail -5

# If deploy <30 min ago, rollback immediately
kubectl rollout undo deployment/mcp-server -n production

# Monitor error rate for 2 minutes
watch -n 5 'curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(mcp_tool_invocations_total{status=\"error\"}[1m]))"'
```

**Option B: Disable problematic tool (if single tool)**
```bash
# Identify failing tool
kubectl logs -n production -l app=mcp-server --tail=500 | grep '"tool":"' | jq -r .tool | sort | uniq -c | sort -rn

# Remove tool from availability (requires server restart)
kubectl set env deployment/mcp-server -n production DISABLED_TOOLS="problematic_tool_name"
```

**Option C: Scale up + Enable circuit breaker**
```bash
# Scale up for redundancy
kubectl scale deployment/mcp-server -n production --replicas=20

# Enable circuit breaker (if external API is down)
kubectl set env deployment/mcp-server -n production CIRCUIT_BREAKER_ENABLED=true
```

### 3. Root Cause Analysis (10 minutes)
Run diagnostics in parallel:

**Check 1: External dependencies**
```bash
# Check if GitHub API is down
curl -w "@curl-format.txt" -o /dev/null -s https://api.github.com/status

# Check database connectivity
kubectl exec -it mcp-server-<pod> -n production -- pg_isready -h db.internal.com

# Check Redis
kubectl exec -it mcp-server-<pod> -n production -- redis-cli -h redis.internal.com ping
```

**Check 2: Resource exhaustion**
```bash
# CPU throttling?
kubectl top pods -n production -l app=mcp-server

# Memory leaks?
kubectl exec -it mcp-server-<pod> -n production -- node -e "console.log(process.memoryUsage())"

# Connection pool exhausted?
kubectl logs -n production -l app=mcp-server --tail=100 | grep "Connection pool"
```

**Check 3: Recent changes**
```bash
# Code changes in last 24h
git log --since="24 hours ago" --oneline

# Infrastructure changes
kubectl get events -n production --sort-by='.lastTimestamp' | head -30

# Config changes
kubectl diff configmap mcp-config -n production
```

### 4. Communication (ongoing)
```bash
# Post to incident channel
slack-cli send --channel=#incidents "SEV2: MCP server error rate 25%. Mitigated via rollback. RCA in progress."

# Update status page
statuspage-cli update --component=ai-assistant --status=degraded --message="AI assistant experiencing high error rates. Team is investigating."
```

### 5. Post-Incident (24h after resolution)
- [ ] Write postmortem (blameless)
- [ ] Identify root cause and contributing factors
- [ ] Create action items with owners and deadlines
- [ ] Update runbook based on learnings
```

**2. Automated Incident Response:**

```typescript
// auto-mitigation.ts
import { Prometheus } from "prometheus-query";
import { KubernetesClient } from "@kubernetes/client-node";

const prometheus = new Prometheus({ endpoint: "http://prometheus:9090" });
const k8s = new KubernetesClient();

interface MitigationAction {
  name: string;
  execute: () => Promise<void>;
  rollback: () => Promise<void>;
}

const mitigations: MitigationAction[] = [
  {
    name: "Rollback to previous version",
    execute: async () => {
      await k8s.rollback("production", "mcp-server");
      console.log("✅ Rolled back deployment");
    },
    rollback: async () => {
      // Rollback of rollback = roll forward
      await k8s.rollout("production", "mcp-server", "redo");
    },
  },
  {
    name: "Scale up replicas",
    execute: async () => {
      const current = await k8s.getReplicas("production", "mcp-server");
      await k8s.scale("production", "mcp-server", current * 2);
      console.log(`✅ Scaled from ${current} to ${current * 2} replicas`);
    },
    rollback: async () => {
      const current = await k8s.getReplicas("production", "mcp-server");
      await k8s.scale("production", "mcp-server", current / 2);
    },
  },
  {
    name: "Enable circuit breaker",
    execute: async () => {
      await k8s.setEnv("production", "mcp-server", {
        CIRCUIT_BREAKER_ENABLED: "true",
      });
      console.log("✅ Enabled circuit breaker");
    },
    rollback: async () => {
      await k8s.setEnv("production", "mcp-server", {
        CIRCUIT_BREAKER_ENABLED: "false",
      });
    },
  },
];

async function detectAndMitigate() {
  // Check error rate
  const errorRate = await prometheus.query(
    `sum(rate(mcp_tool_invocations_total{status="error"}[1m])) / sum(rate(mcp_tool_invocations_total[1m]))`
  );

  const errorRatePercent = parseFloat(errorRate[0].value[1]) * 100;

  if (errorRatePercent < 5) {
    console.log(`✅ Error rate normal: ${errorRatePercent.toFixed(2)}%`);
    return;
  }

  console.error(`🚨 High error rate detected: ${errorRatePercent.toFixed(2)}%`);

  // Check if recent deploy (within 30 minutes)
  const lastDeploy = await k8s.getLastDeployTime("production", "mcp-server");
  const minutesSinceDeploy = (Date.now() - lastDeploy.getTime()) / 1000 / 60;

  if (minutesSinceDeploy < 30) {
    console.log("Recent deploy detected, executing rollback mitigation");
    await mitigations[0].execute(); // Rollback

    // Wait 2 minutes and verify
    await sleep(120000);
    const newErrorRate = await getErrorRate();

    if (newErrorRate < 1) {
      console.log("✅ Mitigation successful");
      await notifySlack("Incident auto-resolved via rollback");
    } else {
      console.error("❌ Mitigation failed, escalating to humans");
      await pageOncall("Auto-mitigation failed, manual intervention required");
    }
  } else {
    // Not a recent deploy, try scaling
    console.log("No recent deploy, scaling up");
    await mitigations[1].execute();
  }
}

// Run every minute
setInterval(detectAndMitigate, 60000);
```

**3. Chaos Engineering (Proactive Testing):**

```yaml
# chaos-experiment.yaml
apiVersion: chaos-mesh.org/v1alpha1
kind: PodChaos
metadata:
  name: mcp-server-kill-test
  namespace: production
spec:
  action: pod-kill
  mode: one
  selector:
    namespaces:
      - production
    labelSelectors:
      app: mcp-server
  scheduler:
    cron: "@weekly"  # Run every week during low-traffic hours
---
apiVersion: chaos-mesh.org/v1alpha1
kind: NetworkChaos
metadata:
  name: github-api-latency
  namespace: production
spec:
  action: delay
  mode: all
  selector:
    namespaces:
      - production
    labelSelectors:
      app: mcp-server
  delay:
    latency: "500ms"
    correlation: "100"
  duration: "5m"
  scheduler:
    cron: "0 2 * * 0"  # 2 AM Sunday
```

---

#### Interview Question: Capacity Planning for Black Friday

**Question:** Your e-commerce company expects 10x traffic on Black Friday (1M concurrent users instead of 100k). Your MCP infrastructure currently handles 100 req/sec. How do you plan capacity?

**Answer:**

**1. Current Capacity Baseline:**
```
- Traffic: 100k users, 100 req/sec average, 300 req/sec peak
- Infrastructure: 10 pods × 3 regions = 30 pods total
- Per-pod capacity: 10 req/sec
- Total capacity: 300 req/sec (matching current peak)
- Utilization: 33% average, 100% peak
```

**2. Black Friday Projection:**
```
- Expected traffic: 1M users, 1000 req/sec average, 3000 req/sec peak
- Required capacity: 3000 req/sec (10x current)
- Required pods: 300 pods (10x current)
- Cost estimate: $50k/month → $500k for Black Friday week
```

**3. Optimization Strategy:**

**A. Load Testing (3 weeks before):**
```bash
# Gradual load test
k6 run --vus 100 --duration 10m load-test.js
k6 run --vus 500 --duration 10m load-test.js
k6 run --vus 2000 --duration 10m load-test.js  # Simulate Black Friday

# Identify bottlenecks
# Found: PostgreSQL max connections (100) exhausted at 800 req/sec
```

**B. Database Optimization:**
```hcl
# Upgrade RDS instance
resource "aws_db_instance" "primary" {
  instance_class = "db.r6g.8xlarge"  # Was: db.r6g.xlarge
  max_connections = 5000  # Was: 100
}

# Add read replicas
resource "aws_db_instance" "replica_3" {
  replicate_source_db = aws_db_instance.primary.arn
}
```

**C. Aggressive Caching:**
```typescript
// Cache product data for 30 minutes (vs 5 minutes normally)
const CACHE_TTL = isBlackFriday() ? 1800 : 300;

// Pre-warm cache with top 1000 products
async function prewarmCache() {
  const topProducts = await db.query("SELECT * FROM products ORDER BY popularity DESC LIMIT 1000");
  for (const product of topProducts) {
    await redis.setex(`product:${product.id}`, CACHE_TTL, JSON.stringify(product));
  }
}
```

**D. Auto-scaling Configuration:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mcp-server-hpa-blackfriday
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: mcp-server
  minReplicas: 50  # Pre-scale from 10 to 50
  maxReplicas: 500  # Allow burst to 500
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60  # Scale up quickly
      policies:
      - type: Percent
        value: 100  # Double pods every minute
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300  # Scale down slowly
      policies:
      - type: Pods
        value: 5  # Remove 5 pods at a time
        periodSeconds: 60
```

**E. Cost Optimization:**
```hcl
# Use Spot instances for burst capacity
resource "aws_eks_node_group" "blackfriday_burst" {
  capacity_type = "SPOT"
  scaling_config {
    desired_size = 0  # Normally 0
    max_size = 200    # Scale up to 200 during Black Friday
    min_size = 0
  }
  
  # Lifecycle: Only active during Black Friday
  lifecycle {
    create_before_destroy = true
  }
}
```

**4. Execution Timeline:**
```
T-3 weeks: Load testing, identify bottlenecks
T-2 weeks: Database upgrades, caching optimizations
T-1 week: Deploy auto-scaling config, pre-warm caches
T-3 days: Pre-scale to minReplicas=50 (burn-in test)
T-1 day: Final load test at 3000 req/sec
Black Friday: Monitor, ready to scale manually if needed
T+1 day: Scale down gradually, analyze metrics
```

**5. Expected Results:**
- ✅ Handle 3000 req/sec peak (10x normal)
- ✅ P95 latency < 2s (maintained SLO)
- ✅ Error rate < 0.5% (within error budget)
- ✅ Cost: $150k for Black Friday week (vs $500k without optimizations)

**Key lesson:** Capacity planning = load testing + incremental scaling + caching + cost optimization.

---

### 🎯 Platform Engineer (6-15 years experience)

You build internal developer platforms and self-service tools. MCP servers should be **easy to deploy** for any team without SRE intervention.

#### Scenario 1: Self-Service MCP Platform

**Context:** Your company has 30 engineering teams. 12 teams want to deploy their own MCP servers. Currently, they file tickets with your team (2-week turnaround). You need to build a self-service platform.

**Requirements:**
1. Teams can deploy MCP servers via UI or CLI
2. Automatic CI/CD pipeline generation
3. Built-in observability (logs, metrics, traces)
4. Cost allocation per team
5. Security guardrails (no production DB access, rate limits)

**Solution:**

**1. Platform API (Backstage.io Integration):**

```yaml
# backstage-template.yaml
apiVersion: scaffolder.backstage.io/v1beta3
kind: Template
metadata:
  name: mcp-server-template
  title: Deploy MCP Server
  description: Self-service MCP server deployment with best practices
spec:
  owner: platform-team
  type: service
  
  parameters:
    - title: Service Configuration
      required:
        - name
        - team
        - environment
      properties:
        name:
          title: Service Name
          type: string
          pattern: '^[a-z0-9-]+$'
          description: Lowercase alphanumeric with hyphens (e.g., 'github-mcp-server')
        team:
          title: Team
          type: string
          enum: [payments, analytics, customer-success, ml-platform]
        environment:
          title: Environment
          type: string
          enum: [development, staging, production]
          default: development
        tools:
          title: Tools to Expose
          type: array
          items:
            type: object
            properties:
              name:
                type: string
              description:
                type: string
              api_endpoint:
                type: string
                format: uri
    
    - title: Resource Configuration
      properties:
        replicas:
          title: Initial Replicas
          type: integer
          minimum: 1
          maximum: 10
          default: 2
        cpu_limit:
          title: CPU Limit
          type: string
          enum: ['500m', '1000m', '2000m']
          default: '1000m'
        memory_limit:
          title: Memory Limit
          type: string
          enum: ['512Mi', '1Gi', '2Gi']
          default: '1Gi'

  steps:
    - id: fetch-template
      name: Fetch MCP Server Template
      action: fetch:template
      input:
        url: ./skeleton
        values:
          name: ${{ parameters.name }}
          team: ${{ parameters.team }}
          environment: ${{ parameters.environment }}
    
    - id: create-repo
      name: Create GitHub Repository
      action: github:repo:create
      input:
        repoUrl: github.com?owner=mycompany&repo=${{ parameters.name }}
        description: MCP Server for ${{ parameters.team }}
    
    - id: create-pipeline
      name: Create CI/CD Pipeline
      action: github:actions:create
      input:
        repoUrl: ${{ steps.create-repo.output.repoUrl }}
        workflowFile: .github/workflows/deploy.yml
    
    - id: provision-infra
      name: Provision Infrastructure
      action: terraform:apply
      input:
        workspace: ${{ parameters.name }}-${{ parameters.environment }}
        variables:
          service_name: ${{ parameters.name }}
          team: ${{ parameters.team }}
          replicas: ${{ parameters.replicas }}
    
    - id: register-service
      name: Register in Service Catalog
      action: catalog:register
      input:
        repoContentsUrl: ${{ steps.create-repo.output.repoUrl }}
        catalogInfoPath: /catalog-info.yaml

  output:
    links:
      - title: Repository
        url: ${{ steps.create-repo.output.remoteUrl }}
      - title: CI/CD Pipeline
        url: ${{ steps.create-repo.output.remoteUrl }}/actions
      - title: Grafana Dashboard
        url: https://grafana.company.com/d/${{ parameters.name }}
      - title: Documentation
        url: ${{ steps.register-service.output.entityRef }}
```

**2. Generated Skeleton (with guardrails):**

```typescript
// skeleton/src/index.ts (auto-generated)
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { withRateLimiter } from "@company/mcp-middleware";
import { withObservability } from "@company/mcp-middleware";
import { withCostTracking } from "@company/mcp-middleware";

const server = new Server(
  {
    name: "{{service_name}}",
    version: "1.0.0",
  },
  {
    capabilities: { tools: {} },
  }
);

// Auto-injected middleware (platform-enforced)
server.use(withRateLimiter({ maxRequestsPerMinute: 100 }));  // Prevent abuse
server.use(withObservability({ team: "{{team}}" }));         // Auto-metrics
server.use(withCostTracking({ team: "{{team}}" }));          // Chargeback

// Tool definitions (filled by user)
server.setRequestHandler("tools/list", async () => {
  return {
    tools: [
      {{#each tools}}
      {
        name: "{{name}}",
        description: "{{description}}",
        inputSchema: { /* ... */ },
      },
      {{/each}}
    ],
  };
});

server.setRequestHandler("tools/call", async (request) => {
  const { name, arguments: args } = request.params;
  
  // Security guardrail: Block direct database access
  if (args.query?.toLowerCase().includes("drop table")) {
    throw new Error("Destructive SQL operations not allowed");
  }
  
  // Call external API (user-defined logic)
  switch (name) {
    {{#each tools}}
    case "{{name}}": {
      const response = await fetch("{{api_endpoint}}", {
        method: "POST",
        body: JSON.stringify(args),
      });
      return { content: [{ type: "text", text: await response.text() }] };
    }
    {{/each}}
  }
});

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("{{service_name}} running");
}

main();
```

**3. Auto-Generated CI/CD Pipeline:**

```yaml
# .github/workflows/deploy.yml (auto-generated)
name: Deploy MCP Server

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Docker image
        run: |
          docker build -t $ECR_REGISTRY/{{service_name}}:$GITHUB_SHA .
          docker push $ECR_REGISTRY/{{service_name}}:$GITHUB_SHA
      
      - name: Security Scan (Trivy)
        run: |
          trivy image --severity HIGH,CRITICAL $ECR_REGISTRY/{{service_name}}:$GITHUB_SHA
      
      - name: Deploy to {{environment}}
        run: |
          kubectl set image deployment/{{service_name}} \
            {{service_name}}=$ECR_REGISTRY/{{service_name}}:$GITHUB_SHA \
            -n {{team}}-{{environment}}
      
      - name: Run Integration Tests
        run: |
          npm run test:integration
      
      - name: Notify Slack
        run: |
          curl -X POST $SLACK_WEBHOOK \
            -d "{'text':'{{service_name}} deployed to {{environment}} by $GITHUB_ACTOR'}"
```

**4. Cost Allocation Dashboard:**

```typescript
// cost-allocation.ts
import { Prometheus } from "prometheus-query";

async function generateCostReport(month: string) {
  const prometheus = new Prometheus({ endpoint: "http://prometheus:9090" });
  
  // Query costs per team
  const cpuCosts = await prometheus.query(
    `sum(avg_over_time(container_cpu_usage_seconds_total[${month}])) by (team) * 0.05`  // $0.05 per CPU-hour
  );
  
  const openaiCosts = await prometheus.query(
    `sum(increase(mcp_openai_cost_dollars[${month}])) by (team)`
  );
  
  const report = cpuCosts.map((entry) => {
    const team = entry.metric.team;
    const cpuCost = parseFloat(entry.value[1]);
    const aiCost = openaiCosts.find((e) => e.metric.team === team)?.value[1] || 0;
    
    return {
      team,
      cpu_cost: cpuCost,
      ai_api_cost: parseFloat(aiCost),
      total_cost: cpuCost + parseFloat(aiCost),
    };
  });
  
  console.table(report);
  
  // Example output:
  // ┌─────────┬──────────────────┬──────────────┬─────────────┬─────────────┐
  // │ (index) │       team       │   cpu_cost   │ ai_api_cost │ total_cost  │
  // ├─────────┼──────────────────┼──────────────┼─────────────┼─────────────┤
  // │    0    │   'payments'     │    1250.50   │   3200.00   │   4450.50   │
  // │    1    │   'analytics'    │     850.25   │    120.50   │    970.75   │
  // │    2    │ 'customer-success'│    620.00   │   1800.00   │   2420.00   │
  // └─────────┴──────────────────┴──────────────┴─────────────┴─────────────┘
}

generateCostReport("30d");
```

**5. Guardrails & Policy Enforcement:**

```typescript
// policy-engine.ts
interface Policy {
  name: string;
  check: (config: any) => { valid: boolean; message?: string };
}

const policies: Policy[] = [
  {
    name: "No production database direct access",
    check: (config) => {
      if (
        config.environment === "production" &&
        config.tools.some((t: any) => t.api_endpoint.includes("postgresql://"))
      ) {
        return {
          valid: false,
          message: "Production MCP servers cannot access databases directly. Use read-replica or caching layer.",
        };
      }
      return { valid: true };
    },
  },
  {
    name: "Rate limits required",
    check: (config) => {
      if (!config.rate_limit || config.rate_limit > 1000) {
        return {
          valid: false,
          message: "Rate limit must be set and <= 1000 req/min",
        };
      }
      return { valid: true };
    },
  },
  {
    name: "Cost budget per team",
    check: (config) => {
      const teamBudget = getTeamBudget(config.team);
      const currentSpend = getCurrentSpend(config.team);
      
      if (currentSpend >= teamBudget * 0.9) {
        return {
          valid: false,
          message: `Team ${config.team} has used 90% of monthly budget ($${currentSpend}/$${teamBudget})`,
        };
      }
      return { valid: true };
    },
  },
];

function validateDeployment(config: any): void {
  for (const policy of policies) {
    const result = policy.check(config);
    if (!result.valid) {
      throw new Error(`Policy violation: ${policy.name}. ${result.message}`);
    }
  }
}
```

---

#### Scenario 2: Golden Path - Standardized MCP Development

**Context:** Teams are deploying MCP servers with inconsistent patterns (no logging, missing health checks, different auth methods). You need to enforce best practices via "golden path."

**Solution:**

**1. MCP SDK Wrapper (Company-specific):**

```typescript
// @company/mcp-sdk/src/index.ts
import { Server as BaseMCPServer } from "@modelcontextprotocol/sdk/server/index.js";
import { trace, context } from "@opentelemetry/api";
import winston from "winston";

export class CompanyMCPServer extends BaseMCPServer {
  private logger: winston.Logger;
  private tracer: any;

  constructor(config: {
    name: string;
    version: string;
    team: string;
  }) {
    super(
      { name: config.name, version: config.version },
      { capabilities: { tools: {} } }
    );

    // Auto-configure logging
    this.logger = winston.createLogger({
      level: "info",
      format: winston.format.json(),
      defaultMeta: {
        service: config.name,
        team: config.team,
      },
      transports: [
        new winston.transports.Console(),
        new winston.transports.File({ filename: `/var/log/mcp/${config.name}.log` }),
      ],
    });

    // Auto-configure tracing
    this.tracer = trace.getTracer(config.name);

    // Auto-register health check
    this.registerHealthCheck();
  }

  private registerHealthCheck() {
    // Expose /health endpoint (if using SSE transport)
    // This is automatically monitored by Kubernetes
  }

  // Override setRequestHandler to add automatic instrumentation
  setRequestHandler(method: string, handler: any) {
    const instrumentedHandler = async (request: any) => {
      const span = this.tracer.startSpan(`mcp.${method}`);
      
      this.logger.info("Request received", {
        method,
        params: request.params,
      });

      try {
        const result = await handler(request);
        span.setStatus({ code: 0 });
        this.logger.info("Request completed", { method });
        return result;
      } catch (error: any) {
        span.setStatus({ code: 2, message: error.message });
        span.recordException(error);
        this.logger.error("Request failed", {
          method,
          error: error.message,
          stack: error.stack,
        });
        throw error;
      } finally {
        span.end();
      }
    };

    super.setRequestHandler(method, instrumentedHandler);
  }
}

// Export convenience functions
export { CompanyMCPServer as Server };
export * from "@modelcontextprotocol/sdk/server/index.js";
```

**Usage by teams:**

```typescript
// Before (vanilla MCP SDK - inconsistent)
import { Server } from "@modelcontextprotocol/sdk/server/index.js";

const server = new Server({ name: "my-server", version: "1.0.0" }, {});
// No logging, no tracing, no standard patterns

// After (company SDK - best practices enforced)
import { Server } from "@company/mcp-sdk";

const server = new Server({
  name: "my-server",
  version: "1.0.0",
  team: "payments",
});
// ✅ Automatic logging, tracing, health checks, metrics
```

**2. Pre-commit Hooks (Enforce Standards):**

```yaml
# .pre-commit-config.yaml (auto-added to all MCP repos)
repos:
  - repo: local
    hooks:
      - id: mcp-lint
        name: MCP Server Linter
        entry: npx @company/mcp-lint
        language: system
        types: [typescript]
        
      - id: security-scan
        name: Security Scan
        entry: npx @company/mcp-security-scan
        language: system
        
      - id: cost-estimate
        name: Cost Estimate
        entry: node scripts/cost-estimate.js
        language: system
```

```typescript
// @company/mcp-lint rules
const rules = [
  {
    name: "must-have-rate-limiter",
    check: (code) => code.includes("withRateLimiter"),
    message: "All MCP servers must use rate limiting middleware",
  },
  {
    name: "must-have-auth",
    check: (code) => code.includes("authenticateUser") || code.includes("validateToken"),
    message: "All tools must validate user authentication",
  },
  {
    name: "no-hardcoded-secrets",
    check: (code) => !/(api_key|password|secret)\s*=\s*["'][^"']+["']/.test(code),
    message: "No hardcoded secrets allowed. Use environment variables.",
  },
];
```

---

#### Interview Question: Multi-Tenancy Architecture

**Question:** Your platform supports 50 teams deploying MCP servers. How do you architect multi-tenancy to ensure isolation, fair resource usage, and security?

**Answer:**

**1. Namespace-per-team (Kubernetes):**
```yaml
# Logical isolation
apiVersion: v1
kind: Namespace
metadata:
  name: team-payments
  labels:
    team: payments
    cost-center: "1234"
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-cross-team-traffic
  namespace: team-payments
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          team: payments  # Only same-team traffic allowed
```

**2. Resource Quotas (Fair Usage):**
```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: team-payments-quota
  namespace: team-payments
spec:
  hard:
    requests.cpu: "20"       # Max 20 CPU cores
    requests.memory: 40Gi    # Max 40GB RAM
    pods: "50"               # Max 50 pods
    services.loadbalancers: "2"  # Max 2 load balancers
```

**3. Pod Security Standards:**
```yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: restricted-mcp-psp
spec:
  privileged: false  # No privileged containers
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  runAsUser:
    rule: MustRunAsNonRoot
  seLinux:
    rule: RunAsAny
  fsGroup:
    rule: RunAsAny
  volumes:
  - 'configMap'
  - 'emptyDir'
  - 'secret'
  # Block: hostPath, hostNetwork
```

**4. Cost Tracking (Chargeback):**
```typescript
// Each namespace has cost labels
const teamCosts = await prometheus.query(`
  sum(
    rate(container_cpu_usage_seconds_total[30d]) * 0.05 +  # CPU cost
    avg_over_time(container_memory_working_set_bytes[30d]) / 1e9 * 0.01  # Memory cost
  ) by (namespace)
`);
```

**5. Access Control (RBAC):**
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: team-payments-admin
  namespace: team-payments
subjects:
- kind: Group
  name: team-payments
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: ClusterRole
  name: admin
  apiGroup: rbac.authorization.k8s.io
# Teams can ONLY manage their own namespace
```

**Trade-offs:**
- ✅ Strong isolation (namespace + network policies)
- ✅ Fair resource usage (quotas)
- ✅ Cost accountability (labels + monitoring)
- ❌ Higher operational overhead (50 namespaces to manage)
- ❌ Cluster-level resources still shared (control plane, DNS)

**Alternative: Cluster-per-team (for large teams):**
- Strongest isolation
- Higher cost ($100/month per cluster)
- Used for teams with compliance requirements (PCI-DSS, HIPAA)

---

### 🎯 Cloud & AI Leader (15+ years experience)

You set technical strategy, manage budgets, and align AI infrastructure with business goals. MCP is a **strategic platform investment** that needs ROI justification.

#### Scenario 1: Build vs Buy vs Partner Decision

**Context:** Your company (Series C, $50M ARR, 200 engineers) wants to standardize on MCP. Three options:

1. **Build in-house:** Custom MCP platform (~$2M, 12 months, 8 engineers)
2. **Buy commercial:** License enterprise MCP platform ($500k/year)
3. **Partner:** Use Anthropic/OpenAI managed MCP ($based on usage, ~$800k/year)

**Your Task:** Present recommendation to exec team with TCO analysis and risk assessment.

**Solution:**

**1. Total Cost of Ownership (3-year horizon):**

| Category | Build | Buy | Partner |
|----------|-------|-----|---------|
| **Year 1** |
| Development | $1.6M (8 eng × $200k) | $0 | $0 |
| License/Usage | $0 | $500k | $800k |
| Infrastructure | $200k | $100k | $50k (managed) |
| Training | $50k | $25k | $10k |
| **Year 1 Total** | **$1.85M** | **$625k** | **$860k** |
| **Year 2** |
| Maintenance | $800k (4 eng) | $0 | $0 |
| License/Usage | $0 | $550k (+10%) | $960k (+20% growth) |
| Infrastructure | $300k (scale) | $150k | $80k |
| **Year 2 Total** | **$1.1M** | **$700k** | **$1.04M** |
| **Year 3** |
| Maintenance | $800k | $0 | $0 |
| License/Usage | $0 | $605k | $1.15M |
| Infrastructure | $400k | $200k | $120k |
| **Year 3 Total** | **$1.2M** | **$805k** | **$1.27M** |
| **3-Year TCO** | **$4.15M** | **$2.13M** | **$3.17M** |

**2. Strategic Analysis:**

```markdown
## Recommendation: Hybrid Approach

### Phase 1 (Months 1-6): Partner
- Start with Anthropic managed MCP
- Validate use cases, gather requirements
- Build internal expertise
- Cost: $400k for 6 months

### Phase 2 (Months 7-18): Buy + Build
- License commercial MCP platform for core infrastructure
- Build custom extensions for company-specific needs
- Migrate from partner to in-house gradually
- Cost: $1.2M (12 months)

### Phase 3 (Months 19+): Primarily Build
- Most workloads on custom platform
- Use partner for specialized/high-value features
- Cost: $900k/year ongoing

### Total 3-Year Cost: $2.9M
**Savings vs Pure Build: $1.25M (30%)**
**Savings vs Pure Partner: $270k (9%)**

## Risk Assessment

| Risk | Build | Buy | Partner | Hybrid |
|------|-------|-----|---------|--------|
| **Vendor lock-in** | Low | Medium | High | Medium |
| **Time to market** | 12mo | 3mo | 1mo | 2mo |
| **Technical debt** | High | Low | None | Medium |
| **Scalability** | Unlimited | Vendor-limited | Unlimited | High |
| **Customization** | Unlimited | Limited | None | High |
| **Expertise required** | 8 eng | 2 eng | 0 eng | 4 eng |

## Business Alignment

### If Business Priority = Speed to Market
→ Recommend: **Partner** (fastest, lowest risk)

### If Business Priority = Cost Optimization
→ Recommend: **Buy** (lowest 3-year TCO)

### If Business Priority = Competitive Differentiation
→ Recommend: **Build** (maximum customization)

### If Business Priority = Balanced
→ Recommend: **Hybrid** (our recommendation)
```

**3. Executive Presentation (Slide Format):**

```markdown
# MCP Platform Strategy

## The Problem
- 12 teams want to deploy AI tools
- Current approach: Manual, 2-week turnaround
- Projected AI usage: 10x growth in 24 months
- Risk: Fragmentation, security gaps, high support burden

## The Solution: Model Context Protocol (MCP)
- Industry standard (Anthropic, OpenAI, Google)
- Enables self-service AI tool deployment
- Reduces time-to-production from weeks to hours
- Built-in security, observability, cost controls

## Financial Impact

### Cost Savings (Year 1)
| Initiative | Annual Savings |
|------------|----------------|
| Reduce manual integrations | $400k (2 FTE) |
| Faster feature velocity | $800k (4x faster) |
| Prevent shadow AI spending | $300k |
| **Total Savings** | **$1.5M** |

### Investment Required
- Year 1: $860k (partner approach)
- **Net ROI: $640k (74% return)**

## Strategic Alignment

✅ **Product Velocity:** Ship AI features 4x faster
✅ **Developer Experience:** Self-service platform
✅ **Security & Compliance:** Centralized governance
✅ **Cost Control:** Visibility + allocation
✅ **Competitive Advantage:** Industry-leading AI integration

## Recommendation
Start with **Partner** (Anthropic managed), transition to **Hybrid** approach over 18 months.

## Next Steps
1. Approve $860k budget for Year 1
2. Hire AI Platform Lead (1 headcount)
3. Pilot with 3 teams (Payments, Analytics, Customer Success)
4. Review at 6-month checkpoint
```

---

#### Scenario 2: AI Governance Framework

**Context:** Your company's AI usage is exploding (50 models, 30 teams, $2M/year AI API costs). Legal/Compliance is concerned about data leakage, hallucinations, and regulatory risk. You need an AI governance framework.

**Your Task:** Design governance policies for MCP-based AI systems.

**Solution:**

```markdown
# AI Governance Framework for MCP Infrastructure

## 1. Data Classification & Access Control

### Data Tiers
| Tier | Examples | MCP Access | Review Cadence |
|------|----------|------------|----------------|
| **Public** | Marketing content, blog posts | Unrestricted | Quarterly |
| **Internal** | Employee directory, Slack history | Auth required | Monthly |
| **Confidential** | Customer PII, financial data | Encryption + audit | Weekly |
| **Restricted** | SSN, medical records, payment cards | Prohibited | N/A |

### Policy: MCP servers CANNOT access Restricted data
- Enforcement: Network policy blocks DB access
- Alternative: De-identified data in read-replica
- Audit: Quarterly review of all data connections

## 2. Model Approval Process

### Tier 1: Pre-approved Models (Self-service)
- GPT-4, Claude 3.5, Gemini 1.5
- Use case: General productivity, code generation
- Approval: None required
- Cost limit: $5k/month per team

### Tier 2: Specialized Models (Manager approval)
- Fine-tuned models, embeddings, vision models
- Use case: Customer-facing features
- Approval: Engineering Manager + Security review
- Cost limit: $25k/month

### Tier 3: Custom Models (Exec approval)
- Proprietary models, external APIs
- Use case: Core product differentiation
- Approval: VP Engineering + Legal + Security
- Cost limit: $100k/month

### Enforcement
```typescript
// mcp-middleware.ts
async function validateModelUsage(modelName: string, team: string) {
  const tier = getModelTier(modelName);
  const approvals = await getApprovals(team, modelName);
  
  if (tier === 1) {
    return true;  // Pre-approved
  } else if (tier === 2 && !approvals.manager) {
    throw new Error("Manager approval required for this model");
  } else if (tier === 3 && (!approvals.vp || !approvals.legal)) {
    throw new Error("Executive + Legal approval required");
  }
  
  return true;
}
```

## 3. Prompt Injection & Security

### Policy: All user inputs must be sanitized
```typescript
// Input sanitization
function sanitizePrompt(userInput: string): string {
  // Remove system prompts
  let cleaned = userInput.replace(/You are|System:|Assistant:/gi, "");
  
  // Remove jailbreak attempts
  cleaned = cleaned.replace(/ignore previous instructions/gi, "");
  cleaned = cleaned.replace(/DAN mode/gi, "");
  
  // Length limit (prevent DOS)
  if (cleaned.length > 10000) {
    throw new Error("Input too long (max 10,000 characters)");
  }
  
  return cleaned;
}

// Output filtering
function filterSensitiveData(output: string): string {
  // Redact SSN
  output = output.replace(/\b\d{3}-\d{2}-\d{4}\b/g, "[SSN REDACTED]");
  
  // Redact credit cards
  output = output.replace(/\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b/g, "[CARD REDACTED]");
  
  return output;
}
```

## 4. Audit & Compliance

### Logging Requirements
All MCP tool calls must log:
```json
{
  "timestamp": "2024-11-20T10:30:00Z",
  "user_id": "user_12345",
  "team": "payments",
  "tool_name": "get_customer_data",
  "data_classification": "Confidential",
  "model_used": "gpt-4-turbo",
  "input_tokens": 150,
  "output_tokens": 450,
  "cost": "$0.0075",
  "approved_use_case": "customer_support_ticket_analysis"
}
```

### Retention Policy
- Public/Internal: 90 days
- Confidential: 7 years (regulatory requirement)
- Audit reviews: Quarterly by InfoSec

## 5. Incident Response

### Trigger Conditions
| Condition | Severity | Response Time |
|-----------|----------|---------------|
| Data leakage detected | SEV1 | Immediate |
| Unapproved model usage | SEV2 | 4 hours |
| Cost spike (>50% above budget) | SEV3 | 24 hours |
| Prompt injection attempt | SEV3 | 24 hours |

### Response Plan
1. Automated: Disable affected MCP server
2. 15 min: Security team investigates
3. 1 hour: Notify affected users
4. 24 hours: Root cause analysis
5. 1 week: Postmortem + policy update

## 6. Cost Governance

### Budget Allocation
- Each team: $10k/month base
- Overages: Require VP approval
- Quarterly review: Reallocate unused budget

### Cost Alerts
```yaml
alerts:
  - name: team_over_budget
    condition: team_monthly_cost > team_budget * 1.1
    action: notify_team_lead
    
  - name: suspicious_usage
    condition: daily_cost > 3 * avg(daily_cost, 7d)
    action: auto_throttle + notify_security
    
  - name: company_burn_rate
    condition: monthly_cost > $200k
    action: notify_cfo + exec_review
```

## 7. Responsible AI Principles

### Bias Detection
- Quarterly audit of model outputs
- Demographic parity testing for customer-facing features
- Mandatory bias training for AI feature owners

### Transparency
- Users must know when interacting with AI
- Disclose model limitations (e.g., "AI-generated, may contain errors")
- Provide human escalation path

### Explainability
- For high-stakes decisions (loan approval, hiring), log reasoning
- Maintain audit trail of tool calls that influenced decision
```

---

#### Interview Question: AI Infrastructure Investment Priorities

**Question:** You have $5M budget for AI infrastructure next year. How do you allocate across: MCP platform, GPU clusters, vector databases, observability, and talent?

**Answer:**

```markdown
# $5M AI Infrastructure Budget Allocation

## Strategic Context
- Company stage: Series C, $50M ARR, 200 engineers
- AI maturity: Early (6 months of AI features)
- Business goal: Ship 10 AI-powered features in 12 months

## Budget Allocation

### 1. MCP Platform ($1.2M - 24%)
**Why:** Foundation for all AI tool integrations
- License/usage: $600k
- 3 platform engineers: $450k
- Infrastructure: $150k

**ROI:** Enable 30 teams to ship AI features independently
**Multiplier effect:** 10x faster AI feature velocity

### 2. Observability & Governance ($800k - 16%)
**Why:** Prevent costly mistakes, ensure compliance
- OpenTelemetry + Datadog: $300k
- Audit & compliance tools: $200k
- 1.5 engineers: $300k

**ROI:** Prevent 1 major incident ($2M customer impact)
**Risk mitigation:** Regulatory compliance (GDPR, SOC2)

### 3. Vector Databases & RAG ($1M - 20%)
**Why:** Required for knowledge retrieval, semantic search
- Pinecone/Weaviate license: $400k
- Storage (S3): $200k
- 2 ML engineers: $400k

**ROI:** Powers 60% of planned AI features (docs search, recommendations)

### 4. GPU Clusters ($1.5M - 30%)
**Why:** Fine-tuning, local inference for cost optimization
- H100 GPUs (on-demand): $900k/year
- Reserved capacity: $400k
- MLOps tooling: $200k

**ROI:** Reduce OpenAI costs by $800k/year (after Year 1)
**Break-even:** 18 months

### 5. Talent & Training ($500k - 10%)
**Why:** Skills gap is #1 blocker
- 2 AI engineers: $400k
- Team training (courses, conferences): $100k

**ROI:** Accelerate entire org's AI adoption

## Alternative: High-Risk, High-Reward Allocation

If business wants **aggressive innovation**:
- Cut MCP to $600k (use partner, not build)
- Cut observability to $400k (basic monitoring only)
- **Increase GPUs to $2.5M** (bet on model differentiation)
- **Increase talent to $1.5M** (5 AI researchers)

**Trade-off:** Higher potential upside (proprietary models), but 2x risk of failure

## Decision Framework

### If Company Priority = Velocity (Ship features fast)
→ **Recommend:** MCP $1.5M, Observability $500k, Talent $1M, GPUs $1M, Vector DB $1M

### If Company Priority = Cost Optimization
→ **Recommend:** MCP $800k (partner), GPUs $2M (break-even faster), Vector DB $800k, Observability $600k, Talent $800k

### If Company Priority = Innovation (Differentiation)
→ **Recommend:** GPUs $2.5M, Talent $1.5M, Vector DB $600k, MCP $600k, Observability $400k

## Our Recommendation: Balanced (as shown above)
**Rationale:** 
- Company is early in AI maturity → need strong foundation (MCP + observability)
- Series C → need to show ROI → focus on shipping features (not research)
- Hedge bets → invest in both platform (MCP) and innovation (GPUs)

## Measurement Plan
- **Month 6 checkpoint:** 5 AI features shipped → continue as-is
- **Month 6 checkpoint:** <3 AI features → shift budget to talent
- **Month 12 checkpoint:** Cost savings >$500k → increase GPU budget
```

---

**Continue to Step 13 (Practical Exercise)?**

