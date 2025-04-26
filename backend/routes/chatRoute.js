const express = require("express");
const router = express.Router();
const axios = require("axios");

router.post("/", async (req, res) => {
  const { question } = req.body;

  try {
    const response = await axios.post("http://localhost:8000/query", {
      question,
    });

    const answer = response.data.answer || "No answer received."; // <-- safe fallback

    res.json({ response: answer }); // <-- this matches what your frontend expects
  } catch (err) {
    console.error("Chat error:", err);
    res
      .status(500)
      .json({ response: "Failed to get response from Python backend" });
  }
});

module.exports = router;
