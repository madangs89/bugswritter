import dotenv from "dotenv";
dotenv.config();
import express from "express";

import cors from "cors";
import upload from "./configs/multer.js";
import splitRouter from "./routes/split.route.js";
import chatRouter from "./routes/chat.routes.js";

const app = express();

app.use(
  cors({
    origin: "*",
  }),
);

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.get("/", (req, res) => {
  res.send(`<h1>Welcome to the server!</h1>
  <p>Use the following endpoints:</p>
  <ul>
    <li><strong>POST /api/data</strong>: Send JSON data to the server.</li>
    <li><strong>GET /api/data</strong>: Retrieve the stored data from the server.</li>
  </ul>`);
});

app.use("/api/extract", splitRouter);

app.use("/api/conversation", chatRouter);

app.listen(3000, () => {
  console.log("Server is running on port 3000");
});
