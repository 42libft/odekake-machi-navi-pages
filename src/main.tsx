import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles.css";

const root = document.getElementById("root");

if (!root) throw new Error("アプリの表示先が見つかりません。");

createRoot(root).render(<App />);
