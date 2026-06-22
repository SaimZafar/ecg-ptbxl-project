import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import { AuthProvider, useAuth } from "./auth";
import { supabase } from "./supabaseClient";
import Landing from "./pages/Landing";
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import Analyze from "./pages/Analyze";
import ProtectedRoute from "./components/ProtectedRoute";

function Nav() {
  const { user } = useAuth();
  return (
    <nav className="topnav">
      <Link to="/" className="brand">ECG<span className="accent">·</span>Scope</Link>
      <div className="navlinks">
        {user ? (
          <>
            <Link to="/analyze">Analyze</Link>
            <button className="linkbtn" onClick={() => supabase.auth.signOut()}>Log out</button>
          </>
        ) : (
          <>
            <Link to="/login">Log in</Link>
            <Link to="/signup" className="navcta">Sign up</Link>
          </>
        )}
      </div>
    </nav>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Nav />
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route path="/analyze" element={<ProtectedRoute><Analyze /></ProtectedRoute>} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}