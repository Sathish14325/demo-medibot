const express = require("express");
const app = express();
const cors = require("cors");
const chatRoute = require("./routes/chatRoute");

app.use(cors());
app.use(express.json());
app.use("/api/chat", chatRoute);

const PORT = 5000;
app.listen(PORT, () =>
  console.log(`🚀 Node.js server running on port ${PORT}`)
);
