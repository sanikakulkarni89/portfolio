import SectionPage from '../components/SectionPage'

export default function Work() {
  return (
    <SectionPage color="#c4a8e8" icon="💻" title="WORK">
      <p className="section-label">// professional experience</p>

      <div className="work-card">
        <div className="work-role">Microservices Architecture Research Assistant</div>
        <div className="work-company">Syracuse University · Full-time</div>
        <div className="work-date">May 2025 — Present · Syracuse, NY</div>
        <ul className="work-bullets">
          <li>Architected event-driven microservices platform using SNS and Eventbridge for asynchronous communication.</li>
          <li>Implemented CQRS and 4 polyglot databases (Amazon RDS, Redis, MongoDB, Neo4j) to support distributed social network operations.</li>
          <li>Built eventual consistency and fault tolerance patterns with CloudWatch and CloudTrail for logging, monitoring and auditability.</li>
        </ul>
        <div className="tags">
          <span className="tag">AWS</span>
          <span className="tag">Java</span>
          <span className="tag">Microservices</span>
          <span className="tag">MongoDB</span>
          <span className="tag">Redis</span>
        </div>
      </div>

      <div className="work-card">
        <div className="work-role">Software Engineer</div>
        <div className="work-company">All In Design Solutions · Full-time</div>
        <div className="work-date">June 2022 — June 2023</div>
        <ul className="work-bullets">
          <li>Developed Java-based automated pipeline management software with Angular frontend processing 200+ client architectural PDFs monthly.</li>
          <li>Built integrated issue tracking platform with real-time logging, role-based commenting, and automated status updates for 50+ stakeholders.</li>
          <li>Engineered dual-interface dashboard on Azure cloud using PostgreSQL and PyPDF, reducing testing workflows by 40-60%.</li>
        </ul>
        <div className="tags">
          <span className="tag">Java</span>
          <span className="tag">Angular</span>
          <span className="tag">PostgreSQL</span>
          <span className="tag">Azure</span>
          <span className="tag">Python</span>
        </div>
      </div>

      <div className="work-card">
        <div className="work-role">Software Developer</div>
        <div className="work-company">Kulkarni Projects · Internship</div>
        <div className="work-date">Jan 2023 — June 2023</div>
        <ul className="work-bullets">
          <li>Developed and executed 8 responsive websites using WordPress and front-end technologies, reducing bounce rates by 31%.</li>
          <li>Managed data architecture with MySQL and MongoDB, improving page load times by 42%.</li>
          <li>Implemented automated testing protocols that reduced post-deployment bugs by 87%.</li>
        </ul>
        <div className="tags">
          <span className="tag">WordPress</span>
          <span className="tag">MySQL</span>
          <span className="tag">MongoDB</span>
          <span className="tag">HTML/CSS</span>
          <span className="tag">JavaScript</span>
        </div>
      </div>

      <div className="work-card">
        <div className="work-role">Software Developer</div>
        <div className="work-company">Wajooba · Summer Internship</div>
        <div className="work-date">May 2022 — Aug 2022</div>
        <ul className="work-bullets">
          <li>Engineered comprehensive payment module integrating dual payment systems across Android and iOS apps.</li>
          <li>Reduced checkout abandonment by 23% and increased transaction completion rate by 17%.</li>
          <li>Partnered with clients through 12 feedback cycles, improving user satisfaction and app ratings from 4.2 to 4.7 stars.</li>
        </ul>
        <div className="tags">
          <span className="tag">Android</span>
          <span className="tag">iOS</span>
          <span className="tag">Payment Integration</span>
          <span className="tag">UI/UX</span>
          <span className="tag">NLP</span>
        </div>
      </div>
    </SectionPage>
  )
}
