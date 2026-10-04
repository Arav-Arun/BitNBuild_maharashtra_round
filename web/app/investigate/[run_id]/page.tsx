import { InvestigatePage } from "../../../components/pages/InvestigatePage";
export default async function Page({params}:{params:Promise<{run_id:string}>}) { const {run_id}=await params; return <InvestigatePage key={run_id} runId={run_id}/>; }
